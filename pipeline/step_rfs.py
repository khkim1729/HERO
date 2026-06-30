"""
pipeline/step_rfs.py

Entry point for RFS (Recurrence-Free Survival) prediction.

Usage:
    python pipeline/step_rfs.py \
        --common-config configs/common_config.yaml \
        --step-config   configs/step_rfs.yaml

Outputs (under step_config.paths.output_dir):
    feature_table.csv       — all extracted features per patient
    oof_predictions.csv     — OOF risk scores + actual RFS time/event
    results.json            — fold C-indexes, mean, std
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hero.prognosis.rfs import (
    build_feature_table,
    select_features,
    train_and_evaluate,
)
from hero.utils.config import load_yaml, resolve_path
from hero.utils.training import ensure_dir, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Step RFS: CoxPH survival prediction")
    parser.add_argument("--common-config", default="configs/common_config.yaml")
    parser.add_argument("--step-config", default="configs/step_rfs.yaml")
    args = parser.parse_args()

    common_cfg = load_yaml(args.common_config)
    step_cfg = load_yaml(args.step_config)

    project_root = Path(common_cfg["paths"]["project_root"]).resolve()

    # --- Resolve paths ---
    mask_dir = Path(step_cfg["paths"]["mask_dir"])

    pet_dir_raw = step_cfg["paths"].get("pet_dir")
    pet_dir = Path(pet_dir_raw) if pet_dir_raw else None

    # clinical_csv: step config overrides common config
    clinical_csv_raw = step_cfg["paths"].get("clinical_csv") or \
                       common_cfg["paths"].get("clinical_csv")
    clinical_csv = resolve_path(project_root, clinical_csv_raw)

    output_dir = ensure_dir(
        project_root / step_cfg["paths"].get("output_dir", "outputs/rfs")
    )

    # --- Build feature table ---
    print("[Step 1] Building feature table ...")
    clin_cfg = step_cfg["clinical"]

    exclude_raw = step_cfg["paths"].get("exclude_cases")
    exclude_cases = resolve_path(project_root, exclude_raw) if exclude_raw else None

    max_cases = step_cfg.get("debug", {}).get("max_cases", None)

    feature_table = build_feature_table(
        mask_dir=mask_dir,
        clinical_csv=clinical_csv,
        pet_dir=pet_dir,
        clinical_cols=clin_cfg.get("feature_cols", []),
        patient_id_col=clin_cfg.get("patient_id_col", "PatientID"),
        time_col=clin_cfg.get("time_col", "RFS"),
        event_col=clin_cfg.get("event_col", "Relapse"),
        exclude_cases=exclude_cases,
        max_cases=max_cases,
    )
    feature_table.to_csv(output_dir / "feature_table.csv", index=False)
    print(f"  Saved → {output_dir / 'feature_table.csv'}")

    # --- Feature selection ---
    print("\n[Step 2] Selecting features ...")
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
        penalizer=step_cfg["model"].get("penalizer", 0.1),
    )
    print(f"  Selected features: {selected}")

    if not selected:
        print("[ERROR] No features passed selection. Adjust thresholds in step_rfs.yaml.")
        sys.exit(1)

    # --- Train & evaluate ---
    print("\n[Step 3] Cross-validated CoxPH training ...")
    model_cfg = step_cfg.get("model", {})
    results = train_and_evaluate(
        df=feature_table,
        feature_cols=selected,
        time_col="rfs_time",
        event_col="rfs_event",
        n_folds=model_cfg.get("n_folds", 5),
        penalizer=model_cfg.get("penalizer", 0.1),
        seed=common_cfg.get("seed", 42),
    )

    # --- Save outputs ---
    results["oof_predictions"].to_csv(
        output_dir / "oof_predictions.csv", index=False
    )

    summary = {
        "selected_features": selected,
        "fold_cindexes": results["fold_cindexes"],
        "mean_cindex": results["mean_cindex"],
        "std_cindex": results["std_cindex"],
    }
    save_json(summary, output_dir / "results.json")

    print(f"\n  Saved → {output_dir / 'oof_predictions.csv'}")
    print(f"  Saved → {output_dir / 'results.json'}")
    print(
        f"\n[Done] Mean C-index: {results['mean_cindex']:.4f} "
        f"± {results['std_cindex']:.4f}"
    )


if __name__ == "__main__":
    main()
