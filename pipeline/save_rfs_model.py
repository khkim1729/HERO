"""
pipeline/save_rfs_model.py

전체 데이터로 WeibullAFT 학습 후 모델 저장.
inference_v2.py에서 로드해서 사용.

Usage:
    python pipeline/save_rfs_model.py \
        --common-config configs/common_config.yaml \
        --step-config   configs/step_rfs.yaml
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter
from sklearn.preprocessing import StandardScaler

from hero.prognosis.rfs import build_feature_table, select_features
from hero.utils.config import load_yaml, resolve_path


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: Path) -> None:
    import json
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-config", default="configs/common_config.yaml")
    parser.add_argument("--step-config", default="configs/step_rfs.yaml")
    parser.add_argument("--output-dir", default="Task/model/rfs")
    args = parser.parse_args()

    common_cfg = load_yaml(args.common_config)
    step_cfg = load_yaml(args.step_config)
    project_root = Path(common_cfg["paths"]["project_root"]).resolve()

    output_dir = ensure_dir(Path(args.output_dir))

    # --- Build feature table (전체 데이터, 제외 없음) ---
    mask_dir = Path(step_cfg["paths"]["mask_dir"])
    pet_dir_raw = step_cfg["paths"].get("pet_dir")
    pet_dir = Path(pet_dir_raw) if pet_dir_raw else None
    clinical_csv_raw = step_cfg["paths"].get("clinical_csv") or \
                       common_cfg["paths"].get("clinical_csv")
    clinical_csv = resolve_path(project_root, clinical_csv_raw)
    clin_cfg = step_cfg["clinical"]

    print("[1] Building feature table (전체 데이터, 제외 없음) ...")
    feature_table = build_feature_table(
        mask_dir=mask_dir,
        clinical_csv=clinical_csv,
        pet_dir=pet_dir,
        clinical_cols=clin_cfg.get("feature_cols", []),
        patient_id_col=clin_cfg.get("patient_id_col", "PatientID"),
        time_col=clin_cfg.get("time_col", "RFS"),
        event_col=clin_cfg.get("event_col", "Relapse"),
        exclude_cases=None,  # 전체 사용
        use_radiomics=False,
    )

    # --- Feature selection ---
    print("[2] Feature selection ...")
    meta_cols = {"patient_id", "rfs_time", "rfs_event"}
    all_feature_cols = [c for c in feature_table.columns if c not in meta_cols]
    fs_cfg = step_cfg.get("feature_selection", {})
    penalizer = step_cfg["model"].get("penalizer", 0.1)

    selected = select_features(
        df=feature_table,
        feature_cols=all_feature_cols,
        time_col="rfs_time",
        event_col="rfs_event",
        min_cindex=fs_cfg.get("min_univariate_cindex", 0.5),
        max_pearson=fs_cfg.get("max_pearson_corr", 0.9),
        penalizer=penalizer,
    )
    print(f"  Selected: {selected}")

    # --- 전체 데이터로 WeibullAFT 학습 ---
    WEIBULL_PENALIZER = 0.5   # multi-seed 실험 최고 성능 (C-index 0.6629)
    print(f"[3] Training WeibullAFT (penalizer={WEIBULL_PENALIZER}) on full dataset ...")
    X = feature_table[selected].fillna(feature_table[selected].median())
    T = feature_table["rfs_time"].values
    E = feature_table["rfs_event"].values

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=selected)

    train_df = X_scaled.copy()
    train_df["rfs_time"] = T
    train_df["rfs_event"] = E

    aft = WeibullAFTFitter(penalizer=WEIBULL_PENALIZER)
    aft.fit(train_df, duration_col="rfs_time", event_col="rfs_event")

    # --- 저장 ---
    with open(output_dir / "weibull_model.pkl", "wb") as f:
        pickle.dump(aft, f)
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save training medians for NaN imputation at inference time
    training_medians = X[selected].median().to_dict()

    save_json({
        "selected_features": selected,
        "model_type": "WeibullAFT",
        "penalizer": WEIBULL_PENALIZER,
        "n_patients": len(feature_table),
        "training_medians": training_medians,
    }, output_dir / "rfs_model_config.json")

    print(f"\n[Done] 저장 완료 → {output_dir}")
    print(f"  - weibull_model.pkl")
    print(f"  - scaler.pkl")
    print(f"  - rfs_model_config.json")


if __name__ == "__main__":
    main()
