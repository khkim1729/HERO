from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def load_model(path: Path):
    package = joblib.load(path)
    model = package["model"]
    feature_columns = package.get("feature_columns") or package.get("metrics", {}).get("feature_columns")
    if feature_columns is None:
        raise RuntimeError(f"feature_columns not found in model package: {path}")
    return package, model, list(feature_columns)


def predict_one(model_path: Path, feature_df: pd.DataFrame, target_name: str) -> dict:
    package, model, feature_columns = load_model(model_path)

    X = feature_df.copy()
    for col in feature_columns:
        if col not in X.columns:
            X[col] = pd.NA

    X = X[feature_columns]

    pred = model.predict(X)[0]

    prob_dict = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = model.classes_
        prob_dict = {str(c): float(p) for c, p in zip(classes, probs)}

    return {
        "target": target_name,
        "model_path": str(model_path),
        "prediction": str(pred),
        "probabilities": prob_dict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer T/N staging using TotalSeg anatomy RF models.")
    parser.add_argument("--feature-csv", required=True)
    parser.add_argument("--t-model", required=True)
    parser.add_argument("--n-model", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    feature_csv = Path(args.feature_csv)
    t_model = Path(args.t_model)
    n_model = Path(args.n_model)
    out_json = Path(args.out_json)

    df = pd.read_csv(feature_csv)
    if len(df) != 1:
        raise RuntimeError(f"Expected one case row, got {len(df)} rows in {feature_csv}")

    result = {
        "feature_csv": str(feature_csv),
        "T_stage": predict_one(t_model, df, "T"),
        "N_stage": predict_one(n_model, df, "N"),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"Wrote prediction: {out_json}")


if __name__ == "__main__":
    main()
