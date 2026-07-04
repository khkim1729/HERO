"""
HECKTOR 2026 Challenge — Inference Entry Point

I/O convention (Grand Challenge):
  Inputs:
    /input/images/ct/*.mha          — CT image
    /input/images/pet/*.mha         — PET image (SUV)
    /input/ehr.json                 — Clinical data

  Outputs:
    /output/images/head-neck-tumor-segmentation/output.mha
        (uint8, labels: 0=background, 1=GTVp, 2=GTVn, CT geometry)
    /output/t-stage.json            — "T2"
    /output/n-stage.json            — "N1"
    /output/rfs.json                — float risk score (higher = worse)

Model weights at /opt/ml/model/:
    nnunet/Dataset502_HECKTOR2026_noGTVp0/
        nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0 … fold_4
    tn_staging/T/T_stage_rf_model.joblib
    tn_staging/N/N_stage_rf_model.joblib
    rfs/coxph_model.pkl
    rfs/scaler.pkl
    rfs/rfs_model_config.json
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import re
import shutil
import subprocess
import tempfile
from glob import glob
from pathlib import Path
from typing import Optional

import SimpleITK
import joblib
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_PATH  = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH  = Path(os.environ.get("MODEL_DIR", "/opt/ml/model"))

DATASET_NAME   = "Dataset502_HECKTOR2026_noGTVp0"
NNUNET_DIR     = MODEL_PATH / "nnunet"
TN_DIR         = MODEL_PATH / "tn_staging"
RFS_DIR        = MODEL_PATH / "rfs"


# ===========================================================================
# Main entry point
# ===========================================================================

def run():
    # 1. Load inputs
    ct_path  = get_image_file(INPUT_PATH / "images/ct")
    pet_path = get_image_file(INPUT_PATH / "images/pet")
    ehr      = load_json(INPUT_PATH / "ehr.json")

    # 2. Segmentation → numpy array (0/1/2)
    segmentation_array = run_segmentation(ct_path, pet_path, ehr)
    write_segmentation(
        location=OUTPUT_PATH / "images/head-neck-tumor-segmentation",
        array=segmentation_array,
        reference_path=ct_path,
    )

    # 3. TN Staging → ("T2", "N1")
    t_stage, n_stage = run_tn_staging(ct_path, pet_path, ehr, segmentation_array)
    write_json(OUTPUT_PATH / "t-stage.json", t_stage)
    write_json(OUTPUT_PATH / "n-stage.json", n_stage)

    # 4. Prognosis → float
    rfs_score = float(run_prognosis(ct_path, pet_path, ehr, segmentation_array, t_stage, n_stage))

    if not np.isfinite(rfs_score):
        print(f"[WARN] Non-finite RFS score {rfs_score}, fallback to 0.0")
        rfs_score = 0.0

    write_json(OUTPUT_PATH / "rfs.json", rfs_score)

    return 0


def _normalize_stage(value, prefix: str, lo: int, hi: int) -> str:
    """
    Normalize model prediction to Grand Challenge-compatible stage string.

    Examples:
      2       -> T2 / N2
      2.0     -> T2 / N2
      "2.0"   -> T2 / N2
      "T2.0"  -> T2
      "N2B"   -> N2
    """
    s = str(value).strip().upper()

    if s.startswith(prefix):
        s = s[1:]

    # Collapse N2A/N2B/N2C-like labels to N2
    s = re.sub(r"^(\d+)[A-Z]$", r"\1", s)

    try:
        n = int(float(s))
    except Exception as e:
        raise ValueError(f"Invalid {prefix}-stage prediction: {value!r}") from e

    if not (lo <= n <= hi):
        raise ValueError(
            f"Invalid {prefix}-stage range: raw={value!r}, normalized={prefix}{n}"
        )

    return f"{prefix}{n}"


# ===========================================================================
# Subtask 1: Segmentation  (nnU-Net v2, 5-fold ensemble)
# ===========================================================================
#nnU-Net fold를 자동 감지하게 수정
def _available_nnunet_folds() -> list[str]:
    trainer_dir = (
        NNUNET_DIR
        / DATASET_NAME
        / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    )

    fold_dirs = sorted(trainer_dir.glob("fold_*"))
    folds = []

    for p in fold_dirs:
        suffix = p.name.replace("fold_", "")
        if suffix.isdigit():
            folds.append(suffix)

    if not folds:
        raise FileNotFoundError(
            f"No nnU-Net folds found under {trainer_dir}. "
            "Expected fold_0, fold_1, ..."
        )

    return folds


def run_segmentation(ct_path, pet_path, ehr) -> np.ndarray:
    """
    Returns numpy array (z, y, x) uint8 with labels 0/1/2.
    nnU-Net expects NIfTI input → convert from .mha, run predict, read back.
    """
    with tempfile.TemporaryDirectory() as tmp_in, \
         tempfile.TemporaryDirectory() as tmp_out:

        case_id = "case"
        # Convert .mha → .nii.gz for nnU-Net
        ct_nii  = Path(tmp_in) / f"{case_id}_0000.nii.gz"
        pet_nii = Path(tmp_in) / f"{case_id}_0001.nii.gz"
        _mha_to_nifti(ct_path,  ct_nii)
        _mha_to_nifti(pet_path, pet_nii)

        env = os.environ.copy()
        env["nnUNet_results"]       = str(NNUNET_DIR)
        env["nnUNet_raw"]           = "/tmp/nnunet_raw"
        env["nnUNet_preprocessed"]  = "/tmp/nnunet_preprocessed"
        # non-root --user pip install → ~/.local/bin 에 설치됨
        env["PATH"] = str(Path.home() / ".local/bin") + ":" + env.get("PATH", "")

        folds = _available_nnunet_folds()
        folds = folds[:1]
        print(f"[Seg] Using nnU-Net folds: {folds}")

           #지금 코드는 무조건 fold 0 1 2 3 4를 쓰고 있는데, 이렇게 하면 fold_0만 있을때의 오류 차
        subprocess.run([
            "nnUNetv2_predict",
            "-i", tmp_in,
            "-o", tmp_out,
            "-d", DATASET_NAME,
            "-c", "3d_fullres",
            "-f", *folds,
            "-npp", "1",
            "-nps", "1",
            "--disable_tta",
        ], env=env, check=True)

        pred_path = Path(tmp_out) / f"{case_id}.nii.gz"
        if not pred_path.exists():
            candidates = list(Path(tmp_out).glob("*.nii.gz"))
            if not candidates:
                raise FileNotFoundError(f"nnU-Net produced no output in {tmp_out}")
            pred_path = candidates[0]

        pred_nii = nib.load(str(pred_path))
        seg_array = np.round(pred_nii.get_fdata()).astype(np.uint8)

    print(f"[Seg] GTVp voxels={int((seg_array==1).sum())}, GTVn voxels={int((seg_array==2).sum())}")
    return seg_array


def _mha_to_nifti(src: str, dst: Path) -> None:
    img = SimpleITK.ReadImage(str(src))
    SimpleITK.WriteImage(img, str(dst))


# ===========================================================================
# Subtask 2: TN Staging  (Random Forest, pred_components features)
# ===========================================================================


def _prepare_ordinal_branch_df(branch: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Prepare one-row dataframe for ordinal T-stage branch."""
    x = df.copy()

    feature_cols = list(branch["feature_cols"])
    num_cols = list(branch.get("num_cols", []))
    cat_cols = list(branch.get("cat_cols", []))

    for c in feature_cols:
        if c not in x.columns:
            x[c] = np.nan

    for c in num_cols:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    for c in cat_cols:
        if c in x.columns:
            x[c] = x[c].astype("object").where(x[c].notna(), "__missing__")

    return x[feature_cols]


def _ordinal_branch_probs(branch: dict, df: pd.DataFrame) -> dict:
    """Return probabilities for T>=2, T>=3, T>=4 for one ordinal branch."""
    x = _prepare_ordinal_branch_df(branch, df)
    x_mat = branch["preprocessor"].transform(x)

    out = {}
    for key in ["T_ge_2", "T_ge_3", "T_ge_4"]:
        clf = branch["classifiers"][key]
        proba = clf.predict_proba(x_mat)

        # Usually classes_ = [0, 1], but handle safely.
        classes = list(getattr(clf, "classes_", [0, 1]))
        if 1 in classes:
            idx = classes.index(1)
        elif True in classes:
            idx = classes.index(True)
        else:
            idx = proba.shape[1] - 1

        out[key] = float(proba[0, idx])

    return out


def _predict_tstage_ordinal(artifact: dict, df: pd.DataFrame) -> str:
    """Predict T1/T2/T3/T4 using ordinal probability ensemble artifact."""
    pet = _ordinal_branch_probs(artifact["branches"]["petaware"], df.copy())
    fin = _ordinal_branch_probs(artifact["branches"]["final"], df.copy())

    w_pet = float(artifact["weights"].get("petaware", 0.8))
    w_fin = float(artifact["weights"].get("final", 0.2))
    denom = w_pet + w_fin
    if denom <= 0:
        w_pet, w_fin, denom = 0.8, 0.2, 1.0

    p2 = (w_pet * pet["T_ge_2"] + w_fin * fin["T_ge_2"]) / denom
    p3 = (w_pet * pet["T_ge_3"] + w_fin * fin["T_ge_3"]) / denom
    p4 = (w_pet * pet["T_ge_4"] + w_fin * fin["T_ge_4"]) / denom

    thr2 = float(artifact["thresholds"]["T_ge_2"])
    thr3 = float(artifact["thresholds"]["T_ge_3"])
    thr4 = float(artifact["thresholds"]["T_ge_4"])

    # Ordinal decision rule:
    # T4 if P(T>=4) passes threshold, else T3, else T2, else T1.
    if p4 >= thr4:
        pred = "T4"
    elif p3 >= thr3:
        pred = "T3"
    elif p2 >= thr2:
        pred = "T2"
    else:
        pred = "T1"

    print(
        "[T-ordinal] "
        f"pet={pet}, final={fin}, "
        f"ensemble={{'T_ge_2': {p2:.4f}, 'T_ge_3': {p3:.4f}, 'T_ge_4': {p4:.4f}}}, "
        f"pred={pred}"
    )

    return pred


def _predict_stage_sklearn_pkg(pkg: dict, df: pd.DataFrame):
    """Predict with existing sklearn RF package format: {'model', 'feature_columns'}."""
    model = pkg["model"]
    cols = list(pkg["feature_columns"])

    x = df.copy()
    for c in cols:
        if c not in x.columns:
            x[c] = np.nan

    return model.predict(x[cols])[0]


def run_tn_staging(ct_path, pet_path, ehr, segmentation_array) -> tuple[str, str]:
    """
    Returns (t_stage, n_stage) e.g. ("T2", "N1").

    T-stage:
        ordinal ensemble artifact with branches/thresholds/weights, if present.
    N-stage:
        existing Random Forest package with model/feature_columns.
    """
    ct_img = SimpleITK.ReadImage(str(ct_path))
    spacing = np.array(ct_img.GetSpacing())  # (sx, sy, sz) in mm

    gtv_feats = _extract_gtv_features(segmentation_array, spacing)
    clin_feats = _parse_ehr_tn(ehr)

    row = {
        "OriginalPatientID": "inference_case",
        "spacing_x_mm": float(spacing[0]),
        "spacing_y_mm": float(spacing[1]),
        "spacing_z_mm": float(spacing[2]),
        "feature_source": "pred_components",
        **clin_feats,
        **gtv_feats,
    }

    df = pd.DataFrame([row])

    t_pkg = joblib.load(TN_DIR / "T" / "T_stage_rf_model.joblib")
    n_pkg = joblib.load(TN_DIR / "N" / "N_stage_rf_model.joblib")

    # T-stage: ordinal artifact or fallback sklearn package.
    if isinstance(t_pkg, dict) and "branches" in t_pkg and "thresholds" in t_pkg and "weights" in t_pkg:
        raw_t = _predict_tstage_ordinal(t_pkg, df.copy())
    else:
        raw_t = _predict_stage_sklearn_pkg(t_pkg, df.copy())

    # N-stage stays as existing RF package.
    raw_n = _predict_stage_sklearn_pkg(n_pkg, df.copy())

    t_stage = _normalize_stage(raw_t, "T", 1, 4)
    n_stage = _normalize_stage(raw_n, "N", 0, 3)

    print(f"[TN] raw T={raw_t!r} -> {t_stage}, raw N={raw_n!r} -> {n_stage}")
    return t_stage, n_stage


def _parse_ehr_tn(ehr: dict) -> dict:
    """Map EHR JSON → clinical_* features used by RF model."""
    def _f(key, default=np.nan):
        v = ehr.get(key)
        if v is None or str(v).strip() in ("", "nan"):
            return default
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    return {
        "clinical_CenterID":            _f("CenterID"),
        "clinical_Age":                 _f("Age"),
        "clinical_Gender":              _f("Gender"),
        "clinical_Tobacco_Consumption": _f("Tobacco Consumption"),
        "clinical_Alcohol_Consumption": _f("Alcohol Consumption"),
        "clinical_Performance_Status":  _f("Performance Status"),
        "clinical_Treatment":           _f("Treatment"),
        "clinical_HPV_Status":          _f("HPV Status"),
        "clinical_Relapse":             np.nan,   # unknown at inference
        "clinical_RFS":                 np.nan,   # unknown at inference
    }


def _extract_gtv_features(mask_arr: np.ndarray, spacing: np.ndarray) -> dict:
    """
    Extract pred_components shape features from segmentation numpy array.
    spacing: (sx, sy, sz) in mm from SimpleITK GetSpacing().
    mask_arr shape: (z, y, x) from nibabel loading of nnU-Net NIfTI output.

    NOTE: nibabel loads NIfTI in (x, y, z) order — shape (512, 512, 90) for typical CT.
    SimpleITK.GetSpacing() returns (sx, sy, sz) matching x, y, z axes.
    So spacing[i] directly corresponds to mask_arr axis i — no reversal needed.
    Training used nibabel affine column-norms which also give (sx, sy, sz) for (x,y,z) array.
    """
    vox_vol = float(np.prod(spacing))

    def _region_stats(binary_mask):
        vox = int(binary_mask.sum())
        if vox == 0:
            return dict(present=0, voxel_count=0, volume_mm3=0.0,
                        bbox_x_mm=0.0, bbox_y_mm=0.0, bbox_z_mm=0.0,
                        bbox_longest_mm=0.0, bbox_shortest_mm=0.0,
                        centroid_x_mm=0.0, centroid_y_mm=0.0, centroid_z_mm=0.0,
                        _centroid=None)
        coords  = np.argwhere(binary_mask)            # (N,3) in (z,y,x) nibabel order
        phys    = coords * spacing                # physical mm using (sz,sy,sx)
        centroid = phys.mean(axis=0)
        bbox_mm  = ((coords.max(0) - coords.min(0) + 1).astype(float)) * spacing
        return dict(present=1, voxel_count=vox, volume_mm3=float(vox * vox_vol),
                    bbox_x_mm=float(bbox_mm[0]), bbox_y_mm=float(bbox_mm[1]),
                    bbox_z_mm=float(bbox_mm[2]),
                    bbox_longest_mm=float(bbox_mm.max()),
                    bbox_shortest_mm=float(bbox_mm.min()),
                    centroid_x_mm=float(centroid[0]),
                    centroid_y_mm=float(centroid[1]),
                    centroid_z_mm=float(centroid[2]),
                    _centroid=centroid)

    gtvp_s = _region_stats(mask_arr == 1)
    gtvn_s = _region_stats(mask_arr == 2)

    feats: dict = {}
    for k, v in gtvp_s.items():
        if k != "_centroid":
            feats[f"gtvp_{k}"] = v
    for k, v in gtvn_s.items():
        if k != "_centroid":
            feats[f"gtvn_total_{k}"] = v

    # Centroid distance
    cp, cn = gtvp_s["_centroid"], gtvn_s["_centroid"]
    feats["gtvp_gtvn_centroid_distance_mm"] = float(np.linalg.norm(cp - cn)) \
        if (cp is not None and cn is not None) else 0.0

    # Volume ratio
    gp_vol = feats["gtvp_volume_mm3"]
    gn_vol = feats["gtvn_total_volume_mm3"]
    feats["gtvn_to_gtvp_volume_ratio"] = gn_vol / gp_vol if gp_vol > 0 else 0.0

    # GTVn connected components
    gtvn_labeled, n_comp = cc_label(mask_arr == 2)
    comp_vols, comp_bbox_longest, comp_centroids = [], [], []
    for c in range(1, n_comp + 1):
        comp  = gtvn_labeled == c
        vol   = float(comp.sum() * vox_vol)
        coords = np.argwhere(comp)
        phys   = coords * spacing
        bbox_mm = ((coords.max(0) - coords.min(0) + 1).astype(float)) * spacing
        comp_vols.append(vol)
        comp_bbox_longest.append(float(bbox_mm.max()))
        comp_centroids.append(phys.mean(0))

    if comp_vols:
        order = np.argsort(comp_vols)[::-1]
        comp_vols         = [comp_vols[i]         for i in order]
        comp_bbox_longest = [comp_bbox_longest[i]  for i in order]
        comp_centroids    = [comp_centroids[i]     for i in order]

    def _nth(lst, n): return lst[n] if len(lst) > n else 0.0

    feats["gtvn_component_count"]                        = n_comp
    feats["gtvn_largest_component_volume_mm3"]           = _nth(comp_vols, 0)
    feats["gtvn_second_largest_component_volume_mm3"]    = _nth(comp_vols, 1)
    feats["gtvn_third_largest_component_volume_mm3"]     = _nth(comp_vols, 2)
    feats["gtvn_largest_component_bbox_longest_mm"]      = _nth(comp_bbox_longest, 0)
    feats["gtvn_second_largest_component_bbox_longest_mm"] = _nth(comp_bbox_longest, 1)
    feats["gtvn_large_component_count_500mm3"]  = sum(1 for v in comp_vols if v > 500)
    feats["gtvn_large_component_count_1000mm3"] = sum(1 for v in comp_vols if v > 1000)
    feats["gtvn_large_component_count_2000mm3"] = sum(1 for v in comp_vols if v > 2000)

    # mask_arr axis 0 = z (nibabel), spacing[0] = sz → z midpoint (matches training)
    img_mid = mask_arr.shape[0] * spacing[0] / 2.0
    left  = sum(1 for c in comp_centroids if c[0] < img_mid)
    right = sum(1 for c in comp_centroids if c[0] >= img_mid)
    feats["gtvn_left_component_count_proxy"]  = left
    feats["gtvn_right_component_count_proxy"] = right
    feats["gtvn_bilateral_proxy"]             = int(left > 0 and right > 0)

    return feats


# ===========================================================================
# Subtask 3: Prognosis  (CoxPH)
# ===========================================================================

def run_prognosis(ct_path, pet_path, ehr, segmentation_array, t_stage, n_stage) -> float:
    """
    Returns float RFS risk score (higher = higher recurrence risk).
    """
    with open(RFS_DIR / "coxph_model.pkl", "rb") as f:
        cph = pickle.load(f)
    with open(RFS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(RFS_DIR / "rfs_model_config.json") as f:
        cfg = json.load(f)
    selected: list[str] = cfg["selected_features"]

    # Get spacing from CT for volume/geometry calculation
    ct_sitk = SimpleITK.ReadImage(str(ct_path))
    spacing = np.array(ct_sitk.GetSpacing())   # (sx, sy, sz) — matches nibabel (x,y,z) axes
    vox_vol = float(np.prod(spacing))

    mask = segmentation_array.astype(np.int32)  # (nx, ny, nz) = (x,y,z) from nibabel
    gtvp_mask = mask == 1
    gtvn_mask = mask == 2

    gtvp_vox     = int(gtvp_mask.sum())
    gtvp_vol_mm3 = gtvp_vox * vox_vol
    gtvp_vol_ml  = gtvp_vol_mm3 / 1000.0

    gtvp_centroid = None
    if gtvp_vox > 0:
        coords = np.argwhere(gtvp_mask)
        gtvp_centroid = (coords * spacing).mean(axis=0)

    gtvn_labeled, gtvn_n = cc_label(gtvn_mask)
    gtvn_vols, gtvn_centroids = [], []
    for c in range(1, gtvn_n + 1):
        comp = gtvn_labeled == c
        gtvn_vols.append(float(comp.sum() * vox_vol))
        gtvn_centroids.append((np.argwhere(comp) * spacing).mean(axis=0))

    gtvn_total_vol_mm3 = sum(gtvn_vols)
    gtvn_total_vol_ml  = gtvn_total_vol_mm3 / 1000.0
    gtvn_max_vol_mm3   = max(gtvn_vols) if gtvn_vols else 0.0

    if gtvp_centroid is not None and gtvn_centroids:
        gtvn_max_dist = float(max(np.linalg.norm(c - gtvp_centroid) for c in gtvn_centroids))
    else:
        gtvn_max_dist = 0.0

    # PET SUV features — resample PET to CT space so shapes match mask (nz,ny,nx)
    pet_feats = {k: np.nan for k in ["gtvp_suv_max", "gtvp_suv_mean", "gtvp_tlg", "gtvn_suv_max"]}
    try:
        pet_sitk_raw = SimpleITK.ReadImage(str(pet_path))
        resampler = SimpleITK.ResampleImageFilter()
        resampler.SetReferenceImage(ct_sitk)
        resampler.SetInterpolator(SimpleITK.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        pet_in_ct = resampler.Execute(pet_sitk_raw)
        # GetArrayFromImage returns (z,y,x); mask from nibabel is (x,y,z) → transpose
        pet_arr = SimpleITK.GetArrayFromImage(pet_in_ct).transpose(2, 1, 0)  # → (x,y,z)
        del pet_sitk_raw, pet_in_ct
        gc.collect()
        if pet_arr.shape == mask.shape:
            gtvp_suv = pet_arr[gtvp_mask]
            if len(gtvp_suv) > 0:
                pet_feats["gtvp_suv_max"]  = float(gtvp_suv.max())
                pet_feats["gtvp_suv_mean"] = float(gtvp_suv.mean())
                pet_feats["gtvp_tlg"]      = float(gtvp_suv.mean() * gtvp_vol_ml)
            gtvn_suv = pet_arr[gtvn_mask]
            if len(gtvn_suv) > 0:
                pet_feats["gtvn_suv_max"] = float(gtvn_suv.max())
        else:
            print(f"[WARN] PET shape {pet_arr.shape} != mask shape {mask.shape} after resampling")
        del pet_arr
        gc.collect()
    except Exception as e:
        print(f"[WARN] PET feature extraction failed: {e}")

    # Clinical features — apply same encoding as training (_encode_clinical in rfs.py):
    #   binary vars (gender, tobacco, alcohol, hpv_status, treatment): 0→-1, 1→1, missing→0
    #   continuous (age, performance_status, center_id): raw value, missing→0 or median
    def _f_raw(key):
        v = ehr.get(key)
        if v is None or str(v).strip() in ("", "nan"):
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    def _encode_binary(key):
        """Map 1→1, 0→-1, missing→0 (matches training _encode_clinical)."""
        v = _f_raw(key)
        if v is None:
            return 0.0
        return 1.0 if v == 1.0 else -1.0

    def _encode_gender(key):
        """Map 1→1, 0→-1, missing→0."""
        v = _f_raw(key)
        if v is None:
            return 0.0
        return 1.0 if v == 1.0 else -1.0

    def _f_continuous(key, missing_val=0.0):
        v = _f_raw(key)
        return v if v is not None else missing_val

    all_feats = {
        "gtvp_volume_mm3":           gtvp_vol_mm3,
        "gtvp_volume_ml":            gtvp_vol_ml,
        "gtvn_count":                gtvn_n,
        "gtvn_total_volume_mm3":     gtvn_total_vol_mm3,
        "gtvn_total_volume_ml":      gtvn_total_vol_ml,
        "gtvn_max_volume_mm3":       gtvn_max_vol_mm3,
        "total_tumor_volume_mm3":    gtvp_vol_mm3 + gtvn_total_vol_mm3,
        "total_tumor_volume_ml":     gtvp_vol_ml  + gtvn_total_vol_ml,
        "gtvn_max_dist_from_gtvp_mm": gtvn_max_dist,
        "age":                _f_continuous("Age"),
        "gender":             _encode_gender("Gender"),
        "tobacco":            _encode_binary("Tobacco Consumption"),
        "alcohol":            _encode_binary("Alcohol Consumption"),
        "performance_status": _f_continuous("Performance Status"),
        "hpv_status":         _encode_binary("HPV Status"),
        "treatment":          _encode_binary("Treatment"),
        "center_id":          _f_continuous("CenterID"),
        **pet_feats,
    }

    # Load training medians from config for NaN imputation
    # (single-row X.median() returns NaN for NaN values — must use training medians)
    training_medians: dict = cfg.get("training_medians", {})

    X = pd.DataFrame([{f: all_feats.get(f, np.nan) for f in selected}])
    for col in selected:
        if pd.isna(X.at[0, col]) and col in training_medians:
            X.at[0, col] = training_medians[col]
    X = X.fillna(0.0)   # fallback: remaining NaN → 0
    X_scaled = pd.DataFrame(scaler.transform(X), columns=selected)

    risk = float(cph.predict_partial_hazard(X_scaled).iloc[0])
    print(f"[Prognosis] RFS risk = {risk:.4f}")
    return risk


# ===========================================================================
# I/O utilities
# ===========================================================================

def get_image_file(location: Path) -> str:
    files = (glob(str(location / "*.mha"))
           + glob(str(location / "*.nii.gz"))
           + glob(str(location / "*.tif"))
           + glob(str(location / "*.tiff")))  # .tiff 입력도 찾게 수정
    if not files:
        raise FileNotFoundError(f"No image file found in {location}")
    return files[0]


def load_json(location: Path) -> dict:
    with open(location) as f:
        return json.load(f)

#write_json()을 strict JSON으로 수정 (NaN, Infinity가 JSON에 들어가려고 할 때 오류 도출가능)
def write_json(location: Path, data) -> None:
    location.parent.mkdir(parents=True, exist_ok=True)
    with open(location, "w") as f:
        json.dump(data, f, indent=2, allow_nan=False)


def write_segmentation(location: Path, array: np.ndarray, reference_path: str) -> None:
    location.mkdir(parents=True, exist_ok=True)
    reference = SimpleITK.ReadImage(reference_path)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    # nibabel gives (x, y, z) order; SimpleITK.GetImageFromArray expects (z, y, x)
    array_zyx = array.transpose(2, 1, 0)
    img = SimpleITK.GetImageFromArray(array_zyx.astype(np.uint8))
    img.CopyInformation(reference)
    SimpleITK.WriteImage(img, str(location / "output.mha"), useCompression=True)


if __name__ == "__main__":
    raise SystemExit(run())
