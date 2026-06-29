from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hero.data.clinical import ClinicalPreprocessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Clinical EDA and preprocessing for HECKTOR 2026.")
    parser.add_argument("--clinical-csv", default="data/sample_5/HECKTOR_2026_training_data.csv")
    parser.add_argument("--output-dir", default="outputs/clinical")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.clinical_csv)
    processor = ClinicalPreprocessor()
    features = processor.fit_transform(df)
    enriched = processor.attach_targets(df)

    transformed = pd.DataFrame(features, columns=processor.feature_names_)
    transformed.insert(0, "PatientID", enriched["PatientID"].values)
    transformed["t_stage_label"] = enriched["t_stage_label"].values
    transformed["n_stage_label"] = enriched["n_stage_label"].values
    transformed["relapse_label"] = enriched["relapse_label"].values
    transformed["rfs_time"] = enriched["rfs_time"].values

    transformed.to_csv(output_dir / "clinical_features_processed.csv", index=False)
    processor.save(output_dir / "clinical_preprocessor.pkl")
    processor.export_summary(df, output_dir / "clinical_summary.json")

    print(f"Saved processed clinical features to {output_dir / 'clinical_features_processed.csv'}")
    print(f"Encoded feature dimension: {features.shape[1]}")


if __name__ == "__main__":
    main()

