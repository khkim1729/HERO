from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from hero.data.clinical import ClinicalPreprocessor


def discover_cases(data_root: str | Path) -> list[dict[str, Any]]:
    data_root = Path(data_root)
    cases = []
    for patient_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        patient_id = patient_dir.name
        ct_path = patient_dir / f"{patient_id}__CT.nii.gz"
        pt_path = patient_dir / f"{patient_id}__PT.nii.gz"
        label_path = patient_dir / f"{patient_id}.nii.gz"
        if ct_path.exists() and pt_path.exists() and label_path.exists():
            cases.append({
                "patient_id": patient_id,
                "image": [str(ct_path), str(pt_path)],
                "label": str(label_path),
                "ct_path": str(ct_path),
                "pt_path": str(pt_path),
            })
    if not cases:
        raise FileNotFoundError(f"No valid patient folders found under {data_root}.")
    return cases


def load_clinical_dataframe(clinical_csv: str | Path) -> pd.DataFrame:
    return pd.read_csv(clinical_csv)


def build_case_records(
    data_root: str | Path,
    clinical_csv: str | Path,
    clinical_processor: ClinicalPreprocessor | None = None,
    cleaned_index_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], ClinicalPreprocessor]:
    cases = discover_cases(data_root)
    case_df = pd.DataFrame(cases)
    clinical_df = load_clinical_dataframe(clinical_csv)
    merged = case_df.merge(clinical_df, left_on="patient_id", right_on="PatientID", how="left")

    if clinical_processor is None:
        clinical_processor = ClinicalPreprocessor()
        features = clinical_processor.fit_transform(merged)
    else:
        features = clinical_processor.transform(merged)

    merged = clinical_processor.attach_targets(merged)
    merged["clinical_features"] = list(features.astype(np.float32))
    merged["staging_features"] = list(features[:, : min(8, features.shape[1])].astype(np.float32))

    keep_patients: set[str] | None = None
    if cleaned_index_path and Path(cleaned_index_path).exists():
        with Path(cleaned_index_path).open("r", encoding="utf-8") as handle:
            cleaned = json.load(handle)
        keep_patients = set(cleaned.get("clean_patient_ids", []))

    records = []
    for _, row in merged.iterrows():
        if keep_patients is not None and row["patient_id"] not in keep_patients:
            continue
        records.append({
            "patient_id": row["patient_id"],
            "image": row["image"],
            "label": row["label"],
            "ct_path": row["ct_path"],
            "pt_path": row["pt_path"],
            "clinical_features": row["clinical_features"],
            "staging_features": row["staging_features"],
            "t_stage": int(row["t_stage_label"]),
            "n_stage": int(row["n_stage_label"]),
            "relapse": int(row["relapse_label"]),
            "rfs_time": float(row["rfs_time"]),
            "rfs_event": int(row["rfs_event"]),
        })
    return records, clinical_processor


def split_folds(records: list[dict[str, Any]], n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(records) < 2:
        raise ValueError("At least two records are required for cross-validation.")
    n_splits = max(2, min(n_splits, len(records)))
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(records))
    return list(kfold.split(indices))

