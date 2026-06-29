from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def normalize_case_id(x: Any) -> str:
    return str(x).strip().replace("-", "_").upper()


def load_exclusions(path: Path) -> set[str]:
    cases = {
        normalize_case_id(line)
        for line in path.read_text().splitlines()
        if line.strip()
    }
    if len(cases) != 48:
        raise RuntimeError(f"Expected 48 excluded cases, found {len(cases)} in {path}")
    return cases


def find_case_col(df: pd.DataFrame) -> str:
    preferred = [
        "PatientID", "patient_id", "patientid", "case_id", "CaseID",
        "case", "ID", "id", "subject_id", "SubjectID"
    ]
    for col in preferred:
        if col in df.columns:
            return col

    for col in df.columns:
        vals = df[col].dropna().astype(str).head(50).tolist()
        if any("-" in v or "_" in v for v in vals):
            if any(v.upper().startswith(("CHUM", "CHUP", "CHUS", "CHUV", "MDA", "HGJ", "HMR", "USZ")) for v in vals):
                return col

    raise RuntimeError(
        "Could not find case ID column. "
        f"Available columns: {df.columns.tolist()}"
    )


def find_target_col(df: pd.DataFrame, target: str) -> str | None:
    if target == "T":
        candidates = [
            "T_stage", "T-stage", "Tstage", "T Stage", "T",
            "t_stage", "t-stage", "tstage"
        ]
    else:
        candidates = [
            "N_stage", "N-stage", "Nstage", "N Stage", "N",
            "n_stage", "n-stage", "nstage"
        ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def clean_label(x: Any) -> str:
    s = str(x).strip()
    s = s.replace(" ", "")
    s = s.replace(".0", "")
    return s


def make_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = []
    categorical_cols = []

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_cols, categorical_cols


def safe_n_splits(y: pd.Series, requested: int) -> int:
    counts = y.value_counts()
    min_count = int(counts.min())
    return max(2, min(requested, min_count))


def train_one_target(
    df: pd.DataFrame,
    case_col: str,
    target_col: str,
    target_name: str,
    output_dir: Path,
    n_estimators: int,
    n_splits_requested: int,
    random_state: int,
) -> dict[str, Any]:
    drop_cols = {
        case_col,
        target_col,
        "_normalized_case_id",
        "split",
        "fold",
        "Fold",
        "dataset",
        "Dataset",
    }

    other_target = "N_stage" if target_name == "T" else "T_stage"
    possible_targets = [
        "T_stage", "T-stage", "Tstage", "T Stage", "T",
        "N_stage", "N-stage", "Nstage", "N Stage", "N",
        "t_stage", "t-stage", "tstage",
        "n_stage", "n-stage", "nstage",
    ]
    for c in possible_targets:
        if c != target_col:
            drop_cols.add(c)

    feature_cols = [c for c in df.columns if c not in drop_cols]

    work = df[[case_col, target_col] + feature_cols].copy()
    work[target_col] = work[target_col].map(clean_label)
    work = work[work[target_col].notna()]
    work = work[work[target_col].astype(str).str.len() > 0]
    work = work[work[target_col].astype(str).str.lower() != "nan"]

    X = work[feature_cols]
    y = work[target_col].astype(str)
    case_ids = work[case_col].astype(str)

    n_splits = safe_n_splits(y, n_splits_requested)

    if len(y.unique()) < 2:
        raise RuntimeError(f"{target_name}: Need at least 2 classes. Found {sorted(y.unique())}")

    preprocessor, numeric_cols, categorical_cols = make_preprocessor(X)

    oof_pred = pd.Series(index=work.index, dtype=object)
    fold_rows = []
    fold_models = []

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        clf = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        random_state=random_state + fold,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        oof_pred.iloc[val_idx] = pred

        fold_metrics = {
            "fold": fold,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "accuracy": float(accuracy_score(y_val, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_val, pred)),
            "macro_f1": float(f1_score(y_val, pred, average="macro")),
        }
        fold_rows.append(fold_metrics)
        fold_models.append(clf)

    oof_df = pd.DataFrame(
        {
            "case_id": case_ids.values,
            "y_true": y.values,
            "y_pred": oof_pred.values,
        }
    )

    labels = sorted(y.unique())
    cm = confusion_matrix(oof_df["y_true"], oof_df["y_pred"], labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])

    metrics = {
        "target": target_name,
        "target_column": target_col,
        "num_cases_after_exclusion": int(len(work)),
        "num_features": int(len(feature_cols)),
        "num_numeric_features": int(len(numeric_cols)),
        "num_categorical_features": int(len(categorical_cols)),
        "n_splits": int(n_splits),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().to_dict().items()},
        "oof_accuracy": float(accuracy_score(oof_df["y_true"], oof_df["y_pred"])),
        "oof_balanced_accuracy": float(balanced_accuracy_score(oof_df["y_true"], oof_df["y_pred"])),
        "oof_macro_f1": float(f1_score(oof_df["y_true"], oof_df["y_pred"], average="macro")),
        "fold_metrics": fold_rows,
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }

    target_dir = output_dir / target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    oof_df.to_csv(target_dir / f"{target_name}_stage_rf_oof_predictions.csv", index=False)
    cm_df.to_csv(target_dir / f"{target_name}_stage_rf_confusion_matrix.csv")
    (target_dir / f"{target_name}_stage_rf_metrics.json").write_text(json.dumps(metrics, indent=2))

    final_preprocessor, _, _ = make_preprocessor(X)
    final_model = Pipeline(
        steps=[
            ("preprocess", final_preprocessor),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    final_model.fit(X, y)

    model_package = {
        "model": final_model,
        "target": target_name,
        "target_column": target_col,
        "case_column": case_col,
        "feature_columns": feature_cols,
        "metrics": metrics,
        "fold_models": fold_models,
    }

    joblib.dump(model_package, target_dir / f"{target_name}_stage_rf_model.joblib")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-csv", required=True)
    parser.add_argument("--exclude-file", default="metadata/exclude_cases_48_tn_staging_only.txt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--case-col", default=None)
    parser.add_argument("--t-col", default=None)
    parser.add_argument("--n-col", default=None)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    feature_csv = Path(args.feature_csv)
    exclude_file = Path(args.exclude_file)
    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(feature_csv)
    original_n = len(df)

    case_col = args.case_col or find_case_col(df)
    exclusions = load_exclusions(exclude_file)

    df["_normalized_case_id"] = df[case_col].map(normalize_case_id)

    removed_df = df[df["_normalized_case_id"].isin(exclusions)].copy()
    clean_df = df[~df["_normalized_case_id"].isin(exclusions)].copy()

    removed_ids = sorted(removed_df[case_col].astype(str).tolist())
    missing_exclusions = sorted(exclusions - set(removed_df["_normalized_case_id"].tolist()))

    if len(removed_df) == 0:
        raise RuntimeError(
            "No cases were removed. Check case ID format and case column."
        )

    clean_df_for_save = clean_df.drop(columns=["_normalized_case_id"])
    clean_df_for_save.to_csv(output_dir / f"{args.run_name}_features_after_exclude48.csv", index=False)
    removed_df.drop(columns=["_normalized_case_id"]).to_csv(output_dir / f"{args.run_name}_removed_cases.csv", index=False)
    (output_dir / f"{args.run_name}_missing_exclusions.txt").write_text(
        "\n".join(missing_exclusions) + ("\n" if missing_exclusions else "")
    )

    t_col = args.t_col or find_target_col(clean_df, "T")
    n_col = args.n_col or find_target_col(clean_df, "N")

    summary = {
        "run_name": args.run_name,
        "feature_csv": str(feature_csv),
        "exclude_file": str(exclude_file),
        "case_column": case_col,
        "num_rows_before_exclusion": int(original_n),
        "num_rows_after_exclusion": int(len(clean_df)),
        "num_removed_rows": int(len(removed_df)),
        "num_requested_exclusions": int(len(exclusions)),
        "num_missing_requested_exclusions": int(len(missing_exclusions)),
        "removed_case_ids": removed_ids,
        "missing_requested_exclusions": missing_exclusions,
        "T_stage_column": t_col,
        "N_stage_column": n_col,
    }

    metrics_all = {}

    if t_col is not None:
        metrics_all["T"] = train_one_target(
            clean_df,
            case_col,
            t_col,
            "T",
            output_dir,
            args.n_estimators,
            args.n_splits,
            args.random_state,
        )
    else:
        print("WARNING: Could not find T-stage column. Skipping T model.")

    if n_col is not None:
        metrics_all["N"] = train_one_target(
            clean_df,
            case_col,
            n_col,
            "N",
            output_dir,
            args.n_estimators,
            args.n_splits,
            args.random_state,
        )
    else:
        print("WARNING: Could not find N-stage column. Skipping N model.")

    summary["metrics"] = metrics_all

    (output_dir / f"{args.run_name}_summary.json").write_text(json.dumps(summary, indent=2))

    print("")
    print("==========================================")
    print("RF TN staging complete")
    print("==========================================")
    print(f"Run name:                  {args.run_name}")
    print(f"Feature CSV:               {feature_csv}")
    print(f"Output dir:                {output_dir}")
    print(f"Case column:               {case_col}")
    print(f"Rows before exclusion:     {original_n}")
    print(f"Rows after exclusion:      {len(clean_df)}")
    print(f"Rows removed:              {len(removed_df)}")
    print(f"Requested exclusions:      {len(exclusions)}")
    print(f"Missing exclusions:        {len(missing_exclusions)}")
    print(f"T column:                  {t_col}")
    print(f"N column:                  {n_col}")

    for target, metrics in metrics_all.items():
        print("")
        print(f"{target}-stage RF")
        print(f"  OOF accuracy:          {metrics['oof_accuracy']:.4f}")
        print(f"  OOF balanced accuracy: {metrics['oof_balanced_accuracy']:.4f}")
        print(f"  OOF macro F1:          {metrics['oof_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
