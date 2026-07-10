"""
pipeline/save_rfs_coxph_full.py

전체 데이터(exclude_cases=None, 48개 제외 없이)로 CoxPH 단일 모델만 학습해서 저장.
HECKTOR2026_HERO_validation_passed/inference.py의 run_prognosis() CoxPH 분기에서
그대로 로드해서 쓸 수 있도록, 기존 rfs_model_config.json과 같은 스키마로 저장한다.

save_rfs_ensemble_model.py를 베이스로, WeibullAFT/앙상블 부분만 빼고
CoxPH 단일 모델만 학습하도록 줄인 버전. exclude_cases=None은 그대로 유지.

Usage:
    cd ~/HERO-pass
    python pipeline/save_rfs_coxph_full.py \
        --common-config configs/common_config.yaml \
        --step-config   configs/step_rfs.yaml \
        --output-dir    HECKTOR2026_HERO_validation_passed/model/rfs

Outputs:
    <output-dir>/coxph_model.pkl
    <output-dir>/scaler.pkl
    <output-dir>/rfs_model_config.json   (model_type: "CoxPH")
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler

from hero.prognosis.rfs import build_feature_table, select_features
from hero.utils.config import load_yaml, resolve_path

PENALIZER = 0.1  # 기존 배포 모델(rfs_model_config.json)의 penalizer=0.1과 동일하게 유지


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-config", default="configs/common_config.yaml")
    parser.add_argument("--step-config",   default="configs/step_rfs.yaml")
    parser.add_argument("--output-dir",    default="HECKTOR2026_HERO_validation_passed/model/rfs")
    args = parser.parse_args()

    common_cfg   = load_yaml(args.common_config)
    step_cfg     = load_yaml(args.step_config)
    project_root = Path(common_cfg["paths"]["project_root"]).resolve()
    output_dir   = ensure_dir(Path(args.output_dir))

    # --- 경로 설정 ---
    mask_dir      = Path(step_cfg["paths"]["mask_dir"])
    pet_dir_raw   = step_cfg["paths"].get("pet_dir")
    pet_dir       = Path(pet_dir_raw) if pet_dir_raw else None
    clinical_csv_raw = (step_cfg["paths"].get("clinical_csv")
                        or common_cfg["paths"].get("clinical_csv"))
    clinical_csv  = resolve_path(project_root, clinical_csv_raw)
    clin_cfg      = step_cfg["clinical"]

    # --- Feature table (전체 데이터, exclude_cases=None) ---
    print("[1] Building feature table (전체 데이터, exclude_cases=None) ...")
    feature_table = build_feature_table(
        mask_dir=mask_dir,
        clinical_csv=clinical_csv,
        pet_dir=pet_dir,
        clinical_cols=clin_cfg.get("feature_cols", []),
        patient_id_col=clin_cfg.get("patient_id_col", "PatientID"),
        time_col=clin_cfg.get("time_col", "RFS"),
        event_col=clin_cfg.get("event_col", "Relapse"),
        exclude_cases=None,  # 48개 제외 없이 전체 사용
        use_radiomics=False,
    )
    print(f"  {len(feature_table)} cases loaded")

    # --- Feature selection ---
    print("[2] Feature selection ...")
    meta_cols = {"patient_id", "rfs_time", "rfs_event"}
    all_feature_cols = [c for c in feature_table.columns if c not in meta_cols]
    fs_cfg = step_cfg.get("feature_selection", {})

    selected = select_features(
        df=feature_table,
        feature_cols=all_feature_cols,
        time_col="rfs_time",
        event_col="rfs_event",
        min_cindex=fs_cfg.get("min_univariate_cindex", 0.5),
        max_pearson=fs_cfg.get("max_pearson_corr", 0.9),
        penalizer=0.1,
    )
    print(f"  Selected ({len(selected)}): {selected}")

    # --- 데이터 준비 ---
    X = feature_table[selected].fillna(feature_table[selected].median())
    T = feature_table["rfs_time"].values
    E = feature_table["rfs_event"].values
    training_medians = feature_table[selected].median().to_dict()

    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=selected)

    train_df = X_scaled.copy()
    train_df["rfs_time"]  = T
    train_df["rfs_event"] = E

    # --- CoxPH 학습 (단일 모델, 앙상블 없음) ---
    print(f"\n[3] Training CoxPH (penalizer={PENALIZER}) on full dataset ...")
    cox = CoxPHFitter(penalizer=PENALIZER)
    cox.fit(train_df, duration_col="rfs_time", event_col="rfs_event")
    cox_train_risk = cox.predict_partial_hazard(X_scaled).values
    print(f"  Train risk  mean={cox_train_risk.mean():.4f}  std={cox_train_risk.std():.4f}")

    try:
        train_cindex = cox.concordance_index_
        print(f"  Training concordance_index_ (in-sample, not OOF): {train_cindex:.4f}")
    except Exception as e:
        print(f"  (concordance_index_ unavailable: {e})")

    # --- 저장 (기존 HECKTOR2026_HERO_validation_passed/model/rfs와 동일 스키마) ---
    print("\n[4] Saving ...")
    with open(output_dir / "coxph_model.pkl", "wb") as f:
        pickle.dump(cox, f)
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    save_json({
        "selected_features": selected,
        "penalizer":         PENALIZER,
        "n_patients":        len(feature_table),
        "training_medians":  training_medians,
        "dropped_features":  [],
        "source_csv":        None,  # build_feature_table()로 직접 계산, CSV 아님
        "model_type":        "CoxPH",
    }, output_dir / "rfs_model_config.json")

    print(f"\n[Done] 저장 완료 → {output_dir}")
    print(f"  coxph_model.pkl / scaler.pkl / rfs_model_config.json")
    print(f"  n_patients={len(feature_table)} (48개 제외 이전 기존 653 대비 확인 필요)")


if __name__ == "__main__":
    main()
