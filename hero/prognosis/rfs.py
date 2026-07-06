"""
hero/prognosis/rfs.py

CoxPH-based Recurrence-Free Survival prediction.

Pipeline:
  1. extract_mask_features()  - GTVp/GTVn volume + PET SUV from nnU-Net OOF mask
  2. build_feature_table()    - merge mask features with clinical CSV
  3. select_features()        - univariate Cox filter + correlation pruning
  4. train_and_evaluate()     - k-fold cross-validated CoxPH, reports C-index
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.ndimage import label as cc_label
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# 1. Feature extraction from mask + PET
# ---------------------------------------------------------------------------

def _voxel_volume_mm3(affine: np.ndarray) -> float:
    """Voxel volume in mm³ from NIfTI affine."""
    voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    return float(np.prod(voxel_sizes))


def _voxel_spacing_mm(affine: np.ndarray) -> np.ndarray:
    """Per-axis voxel spacing in mm."""
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def extract_mask_features(
    mask_path: Path,
    pet_path: Optional[Path] = None,
) -> dict:
    """
    Extract volume and PET-intensity features from one patient's mask.

    Args:
        mask_path : nnU-Net OOF prediction (.nii.gz), labels 0/1/2
        pet_path  : raw PET image (_0001.nii.gz in imagesTr), SUV values

    Returns:
        dict of scalar features
    """
    mask_nii = nib.load(str(mask_path))
    mask = mask_nii.get_fdata().astype(np.int32)
    vox_vol = _voxel_volume_mm3(mask_nii.affine)
    spacing = _voxel_spacing_mm(mask_nii.affine)   # (dx, dy, dz) in mm

    feats: dict = {}

    # --- GTVp (label == 1) ---
    gtvp_mask = mask == 1
    gtvp_vox = int(gtvp_mask.sum())
    gtvp_vol_mm3 = gtvp_vox * vox_vol
    feats["gtvp_volume_mm3"] = gtvp_vol_mm3
    feats["gtvp_volume_ml"] = gtvp_vol_mm3 / 1000.0

    gtvp_centroid: Optional[np.ndarray] = None
    if gtvp_vox > 0:
        coords = np.argwhere(gtvp_mask)
        gtvp_centroid = (coords * spacing).mean(axis=0)   # physical mm

    # --- GTVn (label == 2) ---
    gtvn_mask = mask == 2
    gtvn_labeled, gtvn_n = cc_label(gtvn_mask)
    feats["gtvn_count"] = int(gtvn_n)

    gtvn_vols: list[float] = []
    gtvn_centroids: list[np.ndarray] = []
    for c in range(1, gtvn_n + 1):
        comp = gtvn_labeled == c
        vol = int(comp.sum()) * vox_vol
        gtvn_vols.append(vol)
        coords = np.argwhere(comp)
        gtvn_centroids.append((coords * spacing).mean(axis=0))

    feats["gtvn_total_volume_mm3"] = sum(gtvn_vols) if gtvn_vols else 0.0
    feats["gtvn_total_volume_ml"] = feats["gtvn_total_volume_mm3"] / 1000.0
    feats["gtvn_max_volume_mm3"] = max(gtvn_vols) if gtvn_vols else 0.0
    feats["total_tumor_volume_mm3"] = gtvp_vol_mm3 + feats["gtvn_total_volume_mm3"]
    feats["total_tumor_volume_ml"] = feats["total_tumor_volume_mm3"] / 1000.0

    # GTVn max distance from GTVp centroid
    if gtvp_centroid is not None and gtvn_centroids:
        dists = [np.linalg.norm(c - gtvp_centroid) for c in gtvn_centroids]
        feats["gtvn_max_dist_from_gtvp_mm"] = float(max(dists))
    else:
        feats["gtvn_max_dist_from_gtvp_mm"] = 0.0

    # --- PET features ---
    pet_feats = {
        "gtvp_suv_max": np.nan,
        "gtvp_suv_mean": np.nan,
        "gtvp_tlg": np.nan,
        "gtvn_suv_max": np.nan,
    }
    if pet_path is not None and Path(pet_path).exists():
        pet_nii = nib.load(str(pet_path))
        pet = pet_nii.get_fdata()
        if pet.shape == mask.shape:
            gtvp_suv = pet[gtvp_mask]
            if len(gtvp_suv) > 0:
                pet_feats["gtvp_suv_max"] = float(gtvp_suv.max())
                pet_feats["gtvp_suv_mean"] = float(gtvp_suv.mean())
                pet_feats["gtvp_tlg"] = float(gtvp_suv.mean() * feats["gtvp_volume_ml"])
            gtvn_suv = pet[gtvn_mask]
            if len(gtvn_suv) > 0:
                pet_feats["gtvn_suv_max"] = float(gtvn_suv.max())
        else:
            print(
                f"[WARN] PET shape {pet.shape} != mask shape {mask.shape} "
                f"for {mask_path.stem} — PET features skipped"
            )
    feats.update(pet_feats)

    return feats


# ---------------------------------------------------------------------------
# 2. Build feature table
# ---------------------------------------------------------------------------

# Clinical columns expected in the CSV (configurable via step_rfs.yaml)
_DEFAULT_CLINICAL_COLS = [
    "age", "gender", "tobacco", "alcohol",
    "performance_status", "hpv_status", "treatment",
    "center_id",
]

# Map from possible CSV column names → internal name
_COLUMN_ALIASES: dict[str, str] = {
    "PatientID": "patient_id",
    "patientid": "patient_id",
    "patient_id": "patient_id",
    "Age": "age",
    "Gender": "gender",
    "Tobacco": "tobacco",
    "Tobacco Consumption": "tobacco",
    "Alcohol": "alcohol",
    "Alcohol Consumption": "alcohol",
    "Performance status": "performance_status",
    "Performance Status": "performance_status",
    "PerformanceStatus": "performance_status",
    "HPV status": "hpv_status",
    "HPV Status": "hpv_status",
    "HPVstatus": "hpv_status",
    "Surgery": "surgery",
    "Chemotherapy": "chemotherapy",
    "Treatment": "treatment",
    "CenterID": "center_id",
    "T-stage": "t_stage",
    "N-stage": "n_stage",
    "Relapse": "rfs_event",
    "RFS": "rfs_time",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: _COLUMN_ALIASES[c] for c in df.columns if c in _COLUMN_ALIASES}
    return df.rename(columns=rename)


def _encode_clinical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode clinical variables using the Wang et al. (HECKTOR 2022, 3rd place) strategy:
      - positive/present  →  1
      - negative/absent   → -1
      - missing           →  0  (neutral, distinct from both)

    This matters especially for HPV status where HPV+ vs HPV- vs unknown
    carry very different prognostic signals. Using fillna(0) would conflate
    'negative' with 'missing', corrupting the HPV signal.

    Age uses median imputation (continuous variable).
    T/N-stage uses ordinal encoding (T1=1..T4=4, N0=0..N3=3), missing → -1.
    """
    df = df.copy()

    # Gender: HECKTOR CSV already 0/1 → remap to -1/1, missing → 0
    if "gender" in df.columns:
        df["gender"] = pd.to_numeric(df["gender"], errors="coerce")
        df["gender"] = df["gender"].map({1.0: 1, 0.0: -1}).fillna(0)

    # Binary clinical variables: present=1, absent=-1, missing=0
    for col in ["tobacco", "alcohol", "hpv_status", "surgery", "chemotherapy", "treatment"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            df[col] = s.apply(lambda x: 1 if x == 1 else (-1 if x == 0 else 0))

    # Performance status: ordinal 0–4, missing → 0
    if "performance_status" in df.columns:
        df["performance_status"] = pd.to_numeric(
            df["performance_status"], errors="coerce"
        ).fillna(0)

    # Age: continuous, median imputation
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["age"] = df["age"].fillna(df["age"].median())

    # CenterID: numeric
    if "center_id" in df.columns:
        df["center_id"] = pd.to_numeric(df["center_id"], errors="coerce").fillna(0)

    # T-stage: ordinal (T1=1, T2=2, T3=3, T4=4), missing → 0
    if "t_stage" in df.columns:
        t_map = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
        df["t_stage"] = df["t_stage"].map(t_map).fillna(0).astype(float)

    # N-stage: ordinal (N0=0, N1=1, N2=2, N3=3), missing → -1
    if "n_stage" in df.columns:
        n_map = {"N0": 0, "N1": 1, "N2": 2, "N3": 3}
        df["n_stage"] = df["n_stage"].map(n_map).fillna(-1).astype(float)

    return df


def build_feature_table(
    mask_dir: Path,
    clinical_csv: Path,
    pet_dir: Optional[Path] = None,
    clinical_cols: list[str] = _DEFAULT_CLINICAL_COLS,
    patient_id_col: str = "PatientID",
    time_col: str = "RFS",
    event_col: str = "Relapse",
    exclude_cases: Optional[Path] = None,
    max_cases: Optional[int] = None,
    use_radiomics: bool = False,  # reserved for future use
) -> pd.DataFrame:
    """
    Build a combined feature DataFrame (one row per patient).

    Args:
        mask_dir     : folder with OOF prediction .nii.gz files
        clinical_csv : HECKTOR clinical CSV
        pet_dir      : imagesTr folder (PET = CASE_ID_0001.nii.gz)
        clinical_cols: which clinical columns to include
        patient_id_col / time_col / event_col : CSV column names

    Returns:
        DataFrame with patient_id, features, rfs_time, rfs_event
    """
    mask_dir = Path(mask_dir)
    clinical_csv = Path(clinical_csv)

    # Load and normalise clinical CSV
    clin = pd.read_csv(clinical_csv)
    clin = _normalize_columns(clin)
    clin = clin.rename(columns={
        patient_id_col: "patient_id",
        time_col: "rfs_time",
        event_col: "rfs_event",
    })
    clin = clin.dropna(subset=["rfs_time", "rfs_event"])
    clin["rfs_event"] = clin["rfs_event"].astype(int)
    clin = _encode_clinical(clin)

    # Exclude poor-quality segmentation cases
    if exclude_cases is not None:
        exclude_path = Path(exclude_cases)
        if exclude_path.exists():
            excluded = set(exclude_path.read_text().splitlines())
            # Normalize: CHUM_001 and CHUM-001 both match
            excluded_norm = excluded | {s.replace("_", "-") for s in excluded}
            before = len(clin)
            clin = clin[~clin["patient_id"].isin(excluded_norm)].reset_index(drop=True)
            print(f"[INFO] Excluded {before - len(clin)} cases → {len(clin)} remaining")

    # Keep only needed columns
    keep = ["patient_id", "rfs_time", "rfs_event"] + [
        c for c in clinical_cols if c in clin.columns
    ]
    clin = clin[keep]

    # Collect mask features
    # Normalize: build lookup with both original and hyphen↔underscore variants
    rows = []
    mask_files: dict[str, Path] = {}
    for f in mask_dir.glob("*.nii.gz"):
        stem = f.name.replace(".nii.gz", "")
        mask_files[stem] = f
        mask_files[stem.replace("_", "-")] = f  # CHUM_001 → CHUM-001

    patient_ids = clin["patient_id"].iloc[:max_cases] if max_cases else clin["patient_id"]
    total = len(patient_ids)
    for i, pid in enumerate(patient_ids):
        if pid not in mask_files:
            print(f"[WARN] No mask found for {pid} — skipping")
            continue
        mask_path = mask_files[pid]

        pet_path: Optional[Path] = None
        if pet_dir is not None:
            for pid_variant in [pid, pid.replace("-", "_")]:
                candidate = Path(pet_dir) / f"{pid_variant}_0001.nii.gz"
                if candidate.exists():
                    pet_path = candidate
                    break

        print(f"  [{i+1}/{total}] {pid} ...", end="\r", flush=True)
        try:
            mf = extract_mask_features(mask_path, pet_path)
        except Exception as e:
            print(f"\n[WARN] Feature extraction failed for {pid}: {e}")
            continue

        mf["patient_id"] = pid
        rows.append(mf)

    print(f"  [{total}/{total}] Done.            ")

    if not rows:
        raise RuntimeError("No features extracted — check mask_dir and patient IDs.")

    mask_df = pd.DataFrame(rows)
    merged = clin.merge(mask_df, on="patient_id", how="inner")
    print(
        f"[INFO] Feature table: {len(merged)} patients, "
        f"{len(merged.columns) - 3} features"
    )
    return merged


# ---------------------------------------------------------------------------
# 3. Feature selection
# ---------------------------------------------------------------------------

def select_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    time_col: str = "rfs_time",
    event_col: str = "rfs_event",
    min_cindex: float = 0.5,
    max_pearson: float = 0.9,
    penalizer: float = 0.1,
) -> list[str]:
    """
    Univariate CoxPH filter + Pearson correlation pruning.

    1. For each feature: fit univariate Cox, compute C-index on same data.
       Keep if C-index > min_cindex.
    2. For pairs with |Pearson| > max_pearson: keep higher C-index feature.

    Returns:
        Ordered list of selected feature names.
    """
    X = df[feature_cols].copy()
    T = df[time_col].values
    E = df[event_col].values

    # Drop columns that are entirely NaN or constant
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median())
    constant = X.columns[X.nunique() <= 1]
    X = X.drop(columns=constant)
    feature_cols = list(X.columns)

    # Univariate C-index
    cindexes: dict[str, float] = {}
    for col in feature_cols:
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            tmp = pd.DataFrame({"T": T, "E": E, col: X[col]})
            cph.fit(tmp, duration_col="T", event_col="E")
            risk = cph.predict_partial_hazard(tmp).values
            ci = concordance_index(T, -risk, E)
            # Flip if inverted (feature is protective)
            if ci < 0.5:
                ci = 1 - ci
            cindexes[col] = ci
        except Exception:
            cindexes[col] = 0.5

    kept = [c for c, ci in cindexes.items() if ci > min_cindex]
    print(
        f"[INFO] Univariate filter: {len(kept)}/{len(feature_cols)} features "
        f"pass C-index > {min_cindex}"
    )

    # Correlation pruning
    if len(kept) < 2:
        return kept

    corr = X[kept].corr().abs()
    to_drop: set[str] = set()
    for i, a in enumerate(kept):
        for b in kept[i + 1:]:
            if b in to_drop:
                continue
            if corr.loc[a, b] > max_pearson:
                # Drop the one with lower C-index
                drop = a if cindexes[a] < cindexes[b] else b
                to_drop.add(drop)

    selected = [c for c in kept if c not in to_drop]
    print(
        f"[INFO] Correlation pruning: {len(selected)} features remain "
        f"(|Pearson| threshold {max_pearson})"
    )
    return selected


# ---------------------------------------------------------------------------
# 4. Cross-validated CoxPH training + evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    time_col: str = "rfs_time",
    event_col: str = "rfs_event",
    n_folds: int = 5,
    penalizer: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    k-fold cross-validated CoxPH.

    Returns:
        {
            "fold_cindexes": [...],
            "mean_cindex": float,
            "std_cindex": float,
            "oof_predictions": pd.DataFrame (patient_id, risk_score),
        }
    """
    df = df.copy()
    X_all = df[feature_cols].fillna(df[feature_cols].median())
    T_all = df[time_col].values
    E_all = df[event_col].values
    ids_all = df["patient_id"].values

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_cindexes: list[float] = []
    oof_rows: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
        X_tr = X_all.iloc[train_idx]
        X_val = X_all.iloc[val_idx]
        T_tr, E_tr = T_all[train_idx], E_all[train_idx]
        T_val, E_val = T_all[val_idx], E_all[val_idx]

        # Scale
        scaler = StandardScaler()
        X_tr_s = pd.DataFrame(
            scaler.fit_transform(X_tr), columns=feature_cols
        )
        X_val_s = pd.DataFrame(
            scaler.transform(X_val), columns=feature_cols
        )

        # Fit CoxPH
        train_df = X_tr_s.copy()
        train_df["rfs_time"] = T_tr
        train_df["rfs_event"] = E_tr

        cph = CoxPHFitter(penalizer=penalizer)
        try:
            cph.fit(train_df, duration_col="rfs_time", event_col="rfs_event")
        except Exception as e:
            print(f"[WARN] Fold {fold+1} fit failed: {e}")
            continue

        val_df = X_val_s.copy()
        val_df["rfs_time"] = T_val
        val_df["rfs_event"] = E_val

        risk = cph.predict_partial_hazard(val_df).values
        try:
            ci = concordance_index(T_val, -risk, E_val)
        except ZeroDivisionError:
            print(f"  Fold {fold+1}/{n_folds}: skipped (no admissable pairs in val fold)")
            continue
        fold_cindexes.append(ci)
        print(f"  Fold {fold+1}/{n_folds}: C-index = {ci:.4f}")

        for i, idx in enumerate(val_idx):
            oof_rows.append({
                "patient_id": ids_all[idx],
                "risk_score": float(risk[i]),
                "rfs_time": T_val[i],
                "rfs_event": int(E_val[i]),
                "fold": fold + 1,
            })

    mean_ci = float(np.mean(fold_cindexes)) if fold_cindexes else 0.0
    std_ci = float(np.std(fold_cindexes)) if fold_cindexes else 0.0
    print(f"\n[RESULT] Mean C-index: {mean_ci:.4f} ± {std_ci:.4f}")

    return {
        "fold_cindexes": fold_cindexes,
        "mean_cindex": mean_ci,
        "std_cindex": std_ci,
        "oof_predictions": pd.DataFrame(oof_rows),
    }