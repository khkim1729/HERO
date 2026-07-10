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

# Remove very small predicted tumor components before downstream tasks.
# Applies to both GTVp(label=1) and GTVn(label=2).
MIN_TUMOR_COMPONENT_VOXELS = int(os.environ.get("MIN_TUMOR_COMPONENT_VOXELS", "1000"))


# ===========================================================================
# Main entry point
# ===========================================================================

def run():
    """
    Main inference entry point.

    Failsafe behavior:
      - Always attempts to write the four required outputs.
      - If segmentation fails, writes an all-zero segmentation on the CT grid.
      - If TN staging fails, falls back to T2/N0.
      - If RFS fails, falls back to 0.0.
    """
    import traceback

    def _zero_segmentation_like_ct():
        try:
            ref = SimpleITK.ReadImage(str(ct_path))
            if "_ensure_3d_image" in globals():
                ref = _ensure_3d_image(ref)
            shape_zyx = tuple(reversed(ref.GetSize()))
            print(f"[FAILSAFE] Creating zero segmentation with shape zyx={shape_zyx}")
            return np.zeros(shape_zyx, dtype=np.uint8)
        except Exception as e:
            print(f"[FAILSAFE] Could not create CT-shaped zero segmentation: {e}")
            return np.zeros((1, 1, 1), dtype=np.uint8)

    def _write_zero_segmentation_direct():
        location = OUTPUT_PATH / "images/head-neck-tumor-segmentation"
        location.mkdir(parents=True, exist_ok=True)

        ref = SimpleITK.ReadImage(str(ct_path))
        if "_ensure_3d_image" in globals():
            ref = _ensure_3d_image(ref)

        arr = np.zeros(tuple(reversed(ref.GetSize())), dtype=np.uint8)
        img = SimpleITK.GetImageFromArray(arr)
        img.CopyInformation(ref)

        out_path = location / "output.mha"
        SimpleITK.WriteImage(img, str(out_path), useCompression=True)
        print(f"[FAILSAFE] Wrote zero segmentation directly to {out_path}")

    # 1. Load inputs
    ct_path = get_image_file(INPUT_PATH / "images/ct")
    pet_path = get_image_file(INPUT_PATH / "images/pet")
    ehr = load_json(INPUT_PATH / "ehr.json")

    # 2. Segmentation
    try:
        segmentation_array = run_segmentation(ct_path, pet_path, ehr)
    except Exception as e:
        print("[FAILSAFE] Segmentation failed. Falling back to zero mask.")
        print("[FAILSAFE] Segmentation error:", repr(e))
        traceback.print_exc()
        segmentation_array = _zero_segmentation_like_ct()

    segmentation_array = _remove_small_tumor_components(
        segmentation_array,
        min_voxels=MIN_TUMOR_COMPONENT_VOXELS,
    )

    try:
        write_segmentation(
            location=OUTPUT_PATH / "images/head-neck-tumor-segmentation",
            array=segmentation_array,
            reference_path=ct_path,
        )
    except Exception as e:
        print("[FAILSAFE] write_segmentation failed. Writing direct zero mask.")
        print("[FAILSAFE] write_segmentation error:", repr(e))
        traceback.print_exc()
        segmentation_array = _zero_segmentation_like_ct()
        _write_zero_segmentation_direct()

    # 3. TN Staging
    try:
        t_stage, n_stage = run_tn_staging(ct_path, pet_path, ehr, segmentation_array)
    except Exception as e:
        print("[FAILSAFE] TN staging failed. Falling back to T2/N0.")
        print("[FAILSAFE] TN error:", repr(e))
        traceback.print_exc()
        t_stage, n_stage = "T2", "N0"

    try:
        t_stage = _normalize_stage(t_stage, "T", 0, 4)
    except Exception:
        t_stage = "T2"

    try:
        n_stage = _normalize_stage(n_stage, "N", 0, 3)
    except Exception:
        n_stage = "N0"

    write_json(OUTPUT_PATH / "t-stage.json", t_stage)
    write_json(OUTPUT_PATH / "n-stage.json", n_stage)

    # 4. Prognosis
    try:
        rfs_score = float(run_prognosis(ct_path, pet_path, ehr, segmentation_array, t_stage, n_stage))
    except Exception as e:
        print("[FAILSAFE] Prognosis failed. Falling back to 0.0.")
        print("[FAILSAFE] Prognosis error:", repr(e))
        traceback.print_exc()
        rfs_score = 0.0

    if not np.isfinite(rfs_score):
        print(f"[FAILSAFE] Non-finite RFS score {rfs_score}. Falling back to 0.0.")
        rfs_score = 0.0

    write_json(OUTPUT_PATH / "rfs.json", float(rfs_score))

    print(f"[DONE] Outputs written: T={t_stage}, N={n_stage}, RFS={rfs_score}")
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



def _remove_small_tumor_components(segmentation_array: np.ndarray, min_voxels: int = MIN_TUMOR_COMPONENT_VOXELS) -> np.ndarray:
    """
    Remove tiny predicted tumor connected components.

    Labels:
      0 = background
      1 = GTVp
      2 = GTVn

    Components smaller than min_voxels are set to background.
    This postprocessed mask is used for:
      - output segmentation
      - TN staging features
      - RFS/prognosis features
    """
    arr = np.asarray(segmentation_array).copy()

    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 3:
        print(f"[PP] Small-component removal skipped: expected 3D array, got shape={arr.shape}")
        return arr

    if min_voxels <= 0:
        print(f"[PP] Small-component removal disabled: min_voxels={min_voxels}")
        return arr.astype(np.uint8)

    structure = np.ones((3, 3, 3), dtype=np.uint8)

    total_removed_voxels = 0
    for label_value, label_name in [(1, "GTVp"), (2, "GTVn")]:
        mask = arr == label_value
        labeled, n_comp = cc_label(mask, structure=structure)

        if n_comp == 0:
            print(f"[PP] {label_name}: no components")
            continue

        sizes = np.bincount(labeled.ravel())
        remove_ids = [
            cid for cid in range(1, len(sizes))
            if int(sizes[cid]) < int(min_voxels)
        ]

        kept_ids = [
            cid for cid in range(1, len(sizes))
            if int(sizes[cid]) >= int(min_voxels)
        ]

        removed_voxels = int(sum(int(sizes[cid]) for cid in remove_ids))
        total_removed_voxels += removed_voxels

        for cid in remove_ids:
            arr[labeled == cid] = 0

        print(
            f"[PP] {label_name}: components={n_comp}, "
            f"kept={len(kept_ids)}, removed={len(remove_ids)}, "
            f"removed_voxels={removed_voxels}, min_voxels={min_voxels}"
        )

    print(f"[PP] Small-component removal done: total_removed_voxels={total_removed_voxels}")
    return arr.astype(np.uint8)


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


def _same_sitk_geometry(a, b) -> bool:
    return (
        tuple(a.GetSize()) == tuple(b.GetSize())
        and tuple(round(x, 6) for x in a.GetSpacing()) == tuple(round(x, 6) for x in b.GetSpacing())
        and tuple(round(x, 6) for x in a.GetOrigin()) == tuple(round(x, 6) for x in b.GetOrigin())
        and tuple(round(x, 6) for x in a.GetDirection()) == tuple(round(x, 6) for x in b.GetDirection())
    )


def _resample_to_reference(moving, reference, is_label: bool = False):
    interp = SimpleITK.sitkNearestNeighbor if is_label else SimpleITK.sitkLinear
    return SimpleITK.Resample(
        moving,
        reference,
        SimpleITK.Transform(),
        interp,
        0.0,
        moving.GetPixelID(),
    )


def _ensure_3d_image(img):
    """
    Ensure image is scalar 3D.

    Grand Challenge may provide .tif inputs that SimpleITK reads as:
      - 2D images, or
      - VectorImage/RGB-like images.

    nnU-Net expects scalar 3D images. For 2D images, we add a singleton z
    dimension. For vector images, we use component 0.
    """
    if img.GetNumberOfComponentsPerPixel() > 1:
        print(
            f"[I/O] Input is vector image with "
            f"{img.GetNumberOfComponentsPerPixel()} components; using component 0."
        )
        img = SimpleITK.VectorIndexSelectionCast(img, 0)

    if img.GetDimension() == 3:
        return img

    if img.GetDimension() == 2:
        arr = SimpleITK.GetArrayFromImage(img)  # shape: (y, x)
        arr3 = arr[None, ...]                  # shape: (z=1, y, x)

        out = SimpleITK.GetImageFromArray(arr3)

        spacing2 = img.GetSpacing()
        origin2 = img.GetOrigin()
        direction2 = img.GetDirection()

        out.SetSpacing((float(spacing2[0]), float(spacing2[1]), 1.0))
        out.SetOrigin((float(origin2[0]), float(origin2[1]), 0.0))

        identity3 = (
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        )

        # Extend 2D direction to 3D:
        # [d00 d01]      [d00 d01 0]
        # [d10 d11]  ->  [d10 d11 0]
        #                [ 0   0  1]
        try:
            if len(direction2) == 4:
                d00, d01, d10, d11 = [float(x) for x in direction2]
                direction3 = (
                    d00, d01, 0.0,
                    d10, d11, 0.0,
                    0.0, 0.0, 1.0,
                )

                det = float(np.linalg.det(np.array(direction3).reshape(3, 3)))
                if abs(det) < 1e-8:
                    print(
                        f"[WARN] 2D direction produced singular 3D direction "
                        f"det={det}; using identity."
                    )
                    direction3 = identity3
            else:
                direction3 = identity3

            out.SetDirection(direction3)

        except Exception as e:
            print(f"[WARN] Failed to set extended 3D direction: {e}; using identity.")
            out.SetDirection(identity3)

        print(
            f"[I/O] Converted 2D image to 3D: "
            f"size={out.GetSize()}, spacing={out.GetSpacing()}, direction={out.GetDirection()}"
        )
        return out

    raise ValueError(
        f"Unsupported image dimension: {img.GetDimension()}, "
        f"components={img.GetNumberOfComponentsPerPixel()}"
    )



def _read_sitk_3d(path):
    """Read image and force scalar 3D image for .mha/.tif/.tiff inputs."""
    return _ensure_3d_image(SimpleITK.ReadImage(str(path)))


def _spacing3(img) -> np.ndarray:
    """Return spacing as length-3 array even for 2D TIFF-derived images."""
    spacing = np.array(img.GetSpacing(), dtype=float)

    if spacing.size == 2:
        spacing = np.array([spacing[0], spacing[1], 1.0], dtype=float)
    elif spacing.size == 1:
        spacing = np.array([spacing[0], 1.0, 1.0], dtype=float)
    elif spacing.size > 3:
        spacing = spacing[:3].astype(float)

    return spacing


def run_segmentation(ct_path, pet_path, ehr) -> np.ndarray:
    """
    Run nnU-Net segmentation.

    Current validation experiment:
      - For oversized cases, crop to head-and-neck region first.
      - If cropped volume is still > 100,000,000 voxels, return zero segmentation.
      - Use all available nnU-Net folds as an ensemble.
    """
    ct_img = SimpleITK.ReadImage(str(ct_path))
    pet_img = SimpleITK.ReadImage(str(pet_path))

    ct_img = _ensure_3d_image(ct_img)
    pet_img = _ensure_3d_image(pet_img)

    print(f"[Seg] CT size={ct_img.GetSize()}, spacing={ct_img.GetSpacing()}")
    print(f"[Seg] PET size={pet_img.GetSize()}, spacing={pet_img.GetSpacing()}")

    voxel_count = int(np.prod(ct_img.GetSize()))
    print(f"[Seg] voxel_count={voxel_count:,}")

    ct_orig = None
    crop_z_start = 0
    crop_y_start = 0
    crop_x_start = 0

    # Condition 2 first for oversized cases:
    # crop to head-and-neck region, then apply Condition 1 fallback only if still too large.
    if voxel_count > 100_000_000:
        print("[Seg] Large case — applying head-and-neck crop first (Z=500 + XY tissue bbox).")

        ct_orig = ct_img
        ct_arr = SimpleITK.GetArrayFromImage(ct_img)  # z, y, x

        tissue_mask = ct_arr > -200

        # Z crop: take the superior/head side of body tissue.
        z_indices = np.where(tissue_mask.any(axis=(1, 2)))[0]
        if len(z_indices) > 0:
            z_end = min(ct_arr.shape[0], int(z_indices[-1]) + 1)
            z_start = max(0, z_end - 500)
        else:
            z_end = ct_arr.shape[0]
            z_start = max(0, z_end - 500)

        # XY crop: tissue bounding box with margin.
        xy_tissue = tissue_mask.any(axis=0)
        y_idx = np.where(xy_tissue.any(axis=1))[0]
        x_idx = np.where(xy_tissue.any(axis=0))[0]

        if len(y_idx) > 0 and len(x_idx) > 0:
            y_start = max(0, int(y_idx[0]) - 10)
            y_end = min(ct_arr.shape[1], int(y_idx[-1]) + 11)
            x_start = max(0, int(x_idx[0]) - 10)
            x_end = min(ct_arr.shape[2], int(x_idx[-1]) + 11)
        else:
            y_start, y_end = 0, ct_arr.shape[1]
            x_start, x_end = 0, ct_arr.shape[2]

        ct_arr_crop = ct_arr[z_start:z_end, y_start:y_end, x_start:x_end]
        crop_voxels = int(np.prod(ct_arr_crop.shape))

        print(
            f"[Seg] Cropped Z:[{z_start},{z_end}] "
            f"Y:[{y_start},{y_end}] X:[{x_start},{x_end}] "
            f"shape={ct_arr_crop.shape}, voxels={crop_voxels:,}"
        )

        # Condition 1 fallback after crop.
        if crop_voxels > 100_000_000:
            print(
                f"[Seg] Still too large after crop "
                f"({crop_voxels:,} > 100,000,000); returning zero segmentation."
            )
            return np.zeros(tuple(reversed(ct_orig.GetSize())), dtype=np.uint8)

        crop_z_start = z_start
        crop_y_start = y_start
        crop_x_start = x_start

        ct_crop_img = SimpleITK.GetImageFromArray(ct_arr_crop)
        origin = list(ct_img.GetOrigin())
        spacing = ct_img.GetSpacing()
        direction = ct_img.GetDirection()

        # SimpleITK origin order is x, y, z; numpy crop axes are z, y, x.
        origin[0] = origin[0] + x_start * spacing[0]
        origin[1] = origin[1] + y_start * spacing[1]
        origin[2] = origin[2] + z_start * spacing[2]

        ct_crop_img.SetOrigin(tuple(origin))
        ct_crop_img.SetSpacing(spacing)
        ct_crop_img.SetDirection(direction)

        ct_img = ct_crop_img
        print(f"[Seg] Cropped CT size={ct_img.GetSize()}, spacing={ct_img.GetSpacing()}")

    if not _same_sitk_geometry(ct_img, pet_img):
        print("[Seg] PET geometry differs from CT. Resampling PET to CT grid.")
        pet_img = _resample_to_reference(pet_img, ct_img, is_label=False)
        print(f"[Seg] PET resampled size={pet_img.GetSize()}, spacing={pet_img.GetSpacing()}")

    with tempfile.TemporaryDirectory() as tmp_in, tempfile.TemporaryDirectory() as tmp_out:
        tmp_in = Path(tmp_in)
        tmp_out = Path(tmp_out)

        ct_nii = tmp_in / "case_0000.nii.gz"
        pet_nii = tmp_in / "case_0001.nii.gz"

        SimpleITK.WriteImage(ct_img, str(ct_nii))
        SimpleITK.WriteImage(pet_img, str(pet_nii))

        folds = _available_nnunet_folds()
        print(f"[Seg] Using nnU-Net folds: {folds}")

        env = os.environ.copy()
        env["nnUNet_results"] = str(MODEL_PATH / "nnunet")
        env["nnUNet_raw"] = "/tmp/nnUNet_raw"
        env["nnUNet_preprocessed"] = "/tmp/nnUNet_preprocessed"
        env["PATH"] = str(Path.home() / ".local/bin") + ":" + env.get("PATH", "")

        nnunet_cmd = str(Path.home() / ".local/bin" / "nnUNetv2_predict")
        if not Path(nnunet_cmd).exists():
            nnunet_cmd = "nnUNetv2_predict"

        subprocess.run([
            nnunet_cmd,
            "-i", str(tmp_in),
            "-o", str(tmp_out),
            "-d", DATASET_NAME,
            "-c", "3d_fullres",
            "-f", *folds,
            "-npp", "1",
            "-nps", "1",
            "--disable_tta",
            "--not_on_device",
        ], env=env, check=True)

        pred_files = sorted(tmp_out.glob("*.nii.gz"))
        if not pred_files:
            raise FileNotFoundError(f"No nnU-Net prediction found in {tmp_out}")

        pred_img = SimpleITK.ReadImage(str(pred_files[0]))

        if not _same_sitk_geometry(pred_img, ct_img):
            print("[Seg] Prediction geometry differs from CT. Resampling label to CT grid.")
            pred_img = _resample_to_reference(pred_img, ct_img, is_label=True)

        seg = SimpleITK.GetArrayFromImage(pred_img).astype(np.uint8)

    # Paste-back for cropped cases.
    if ct_orig is not None:
        orig_shape = tuple(reversed(ct_orig.GetSize()))  # z, y, x
        full_seg = np.zeros(orig_shape, dtype=np.uint8)

        z_end = crop_z_start + seg.shape[0]
        y_end = crop_y_start + seg.shape[1]
        x_end = crop_x_start + seg.shape[2]

        if z_end > full_seg.shape[0] or y_end > full_seg.shape[1] or x_end > full_seg.shape[2]:
            raise ValueError(
                f"Cropped segmentation paste-back out of bounds: "
                f"seg={seg.shape}, start={(crop_z_start, crop_y_start, crop_x_start)}, "
                f"full={full_seg.shape}"
            )

        full_seg[
            crop_z_start:z_end,
            crop_y_start:y_end,
            crop_x_start:x_end,
        ] = seg

        seg = full_seg
        print(f"[Seg] Pasted back to full CT shape={seg.shape}")

    print(f"[Seg] output segmentation shape={seg.shape}, labels={np.unique(seg).tolist()}")
    return seg

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



def _predict_nstage_ordinal(artifact: dict, df: pd.DataFrame) -> str:
    """Predict N0/N1/N2/N3 using ordinal probability artifact."""
    cols = list(artifact["feature_columns"])

    x = df.copy()
    for c in cols:
        if c not in x.columns:
            x[c] = np.nan

    x = x[cols].copy()

    for c in artifact.get("num_cols", []):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    for c in artifact.get("cat_cols", []):
        if c in x.columns:
            x[c] = x[c].astype("object").where(x[c].notna(), "__missing__")

    x_mat = artifact["preprocessor"].transform(x)

    probs = {}
    for key in ["N_ge_1", "N_ge_2", "N_ge_3"]:
        clf = artifact["classifiers"][key]
        proba = clf.predict_proba(x_mat)
        classes = list(getattr(clf, "classes_", [0, 1]))
        idx = classes.index(1) if 1 in classes else proba.shape[1] - 1
        probs[key] = float(proba[0, idx])

    thresholds = artifact.get("thresholds", {})
    thr1 = float(thresholds.get("N_ge_1", 0.5))
    thr2 = float(thresholds.get("N_ge_2", 0.5))
    thr3 = float(thresholds.get("N_ge_3", 0.5))

    if probs["N_ge_3"] >= thr3:
        pred = "N3"
    elif probs["N_ge_2"] >= thr2:
        pred = "N2"
    elif probs["N_ge_1"] >= thr1:
        pred = "N1"
    else:
        pred = "N0"

    print(
        "[N-ordinal] "
        f"probs={{'N_ge_1': {probs['N_ge_1']:.4f}, "
        f"'N_ge_2': {probs['N_ge_2']:.4f}, "
        f"'N_ge_3': {probs['N_ge_3']:.4f}}}, pred={pred}"
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
    ct_img = _read_sitk_3d(ct_path)
    spacing = _spacing3(ct_img)

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

    # N-stage: ordinal artifact or fallback sklearn package.
    if (
        isinstance(n_pkg, dict)
        and n_pkg.get("model_name") == "nstage_ordinal_rf_probability"
        and "classifiers" in n_pkg
        and "thresholds" in n_pkg
        and "preprocessor" in n_pkg
    ):
        raw_n = _predict_nstage_ordinal(n_pkg, df.copy())
    else:
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
    with open(RFS_DIR / "rfs_model_config.json") as f:
        cfg = json.load(f)

    selected: list[str] = cfg["selected_features"]
    model_type = str(cfg.get("model_type", "CoxPH")).lower()

    if model_type in {"weibullaft", "weibull_aft", "weibull"}:
        with open(RFS_DIR / "weibull_model.pkl", "rb") as f:
            rfs_model = pickle.load(f)
        print("[Prognosis] Loaded WeibullAFT model")
    else:
        with open(RFS_DIR / "coxph_model.pkl", "rb") as f:
            rfs_model = pickle.load(f)
        print("[Prognosis] Loaded CoxPH model")

    with open(RFS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)


    # Get spacing from CT for volume/geometry calculation
    ct_sitk = _read_sitk_3d(ct_path)
    spacing = _spacing3(ct_sitk)
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
        pet_sitk_raw = _read_sitk_3d(pet_path)
        resampler = SimpleITK.ResampleImageFilter()
        resampler.SetReferenceImage(ct_sitk)
        resampler.SetInterpolator(SimpleITK.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        pet_in_ct = resampler.Execute(pet_sitk_raw)
        # GetArrayFromImage returns (z,y,x); mask from nibabel is (x,y,z) → transpose
        pet_arr = SimpleITK.GetArrayFromImage(pet_in_ct)  # (z,y,x), matches segmentation mask
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
    #   continuous (age, performance_status, center_id): raw value, missing -> np.nan so training_medians can impute
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

    def _f_continuous(key):
        v = _f_raw(key)
        return v if v is not None else np.nan

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

    if model_type in {"weibullaft", "weibull_aft", "weibull"}:
        # WeibullAFT predicts survival time. Convert to positive risk score
        # so that higher value still means worse prognosis, consistent with CoxPH partial hazard.
        try:
            pred_time = float(rfs_model.predict_median(X_scaled).iloc[0])
            pred_kind = "median"
        except Exception as e:
            print(f"[WARN] WeibullAFT predict_median failed: {e}; using predict_expectation")
            pred_time = float(rfs_model.predict_expectation(X_scaled).iloc[0])
            pred_kind = "expectation"

        if not np.isfinite(pred_time) or pred_time <= 0:
            print(f"[WARN] Invalid WeibullAFT predicted {pred_kind} survival={pred_time}; fallback risk=0.0")
            risk = 0.0
        else:
            risk = 1.0 / pred_time

        print(f"[Prognosis] WeibullAFT predicted_{pred_kind}_survival={pred_time:.4f}, risk={risk:.6f}")

    else:
        risk = float(rfs_model.predict_partial_hazard(X_scaled).iloc[0])
        print(f"[Prognosis] CoxPH RFS risk = {risk:.6f}")

    return float(risk)


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

    reference = SimpleITK.ReadImage(str(reference_path))
    reference = _ensure_3d_image(reference)

    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]

    arr = array.astype(np.uint8)

    # SimpleITK.GetArrayFromImage gives (z, y, x).
    expected_zyx = tuple(reversed(reference.GetSize()))
    expected_xyz = tuple(reference.GetSize())

    print(f"[I/O] write_segmentation array shape={arr.shape}")
    print(f"[I/O] reference size xyz={reference.GetSize()}, expected zyx={expected_zyx}")

    if tuple(arr.shape) == expected_zyx:
        arr_zyx = arr
    elif tuple(arr.shape) == expected_xyz:
        arr_zyx = arr.transpose(2, 1, 0)
    else:
        raise ValueError(
            f"Segmentation shape {arr.shape} does not match reference. "
            f"Expected zyx={expected_zyx} or xyz={expected_xyz}."
        )

    img = SimpleITK.GetImageFromArray(arr_zyx.astype(np.uint8))
    img.CopyInformation(reference)

    out_path = location / "output.mha"
    SimpleITK.WriteImage(img, str(out_path), useCompression=True)
    print(f"[I/O] Wrote segmentation to {out_path}")



if __name__ == "__main__":
    raise SystemExit(run())
