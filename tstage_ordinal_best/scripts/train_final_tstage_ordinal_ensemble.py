import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LABELS = ["T1", "T2", "T3", "T4"]
STAGE_TO_INT = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}

HU_COLS = [
    "bone_contact_volume_ml",
    "bone_contact_ratio",
    "tumor_inside_bone_envelope_volume_ml",
    "cortical_defect_volume_ml",
    "hu_bone_cue_score_v2",
    "suspicious_bone_invasion_v2",
]


def norm_id(x):
    s = str(x)
    s = s.replace(".nii.gz", "")
    s = s.replace("_0000", "")
    s = s.replace("_0001", "")
    return s


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def ensure_hu_features(df):
    df = df.copy()

    if "suspicious_bone_invasion_v2" not in df.columns:
        df["suspicious_bone_invasion_v2"] = (
            (df["bone_contact_volume_ml"] > 0.10)
            & (
                (df["tumor_inside_bone_envelope_volume_ml"] > 0.05)
                | (df["cortical_defect_volume_ml"] > 0.03)
            )
        ).astype(int)

    if "hu_bone_cue_score_v2" not in df.columns:
        contact_score = (df["bone_contact_volume_ml"] / 0.10).clip(0, 5)
        inside_score = (
            df["tumor_inside_bone_envelope_volume_ml"] / 0.05
        ).clip(0, 5)
        defect_score = (df["cortical_defect_volume_ml"] / 0.03).clip(0, 5)
        df["hu_bone_cue_score_v2"] = contact_score * np.maximum(
            inside_score,
            defect_score,
        )

    return df


def load_base(path):
    df = pd.read_csv(path)

    if "PatientID" not in df.columns:
        raise ValueError("base_features must contain PatientID")

    if "T_stage" not in df.columns:
        raise ValueError("base_features must contain T_stage for training")

    df["PatientID"] = df["PatientID"].map(norm_id)
    df["T_stage"] = df["T_stage"].astype(str)
    df = df[df["T_stage"].isin(LABELS)].copy()
    df = df.reset_index(drop=True)
    return df


def load_hu(path):
    df = ensure_hu_features(pd.read_csv(path))

    if "PatientID" not in df.columns:
        if "case_id" not in df.columns:
            raise ValueError("HU table must contain PatientID or case_id")
        df["PatientID"] = df["case_id"].map(norm_id)
    else:
        df["PatientID"] = df["PatientID"].map(norm_id)

    keep = ["PatientID"] + [c for c in HU_COLS if c in df.columns]
    return df[keep].copy()


def load_geom(path):
    df = pd.read_csv(path)

    if "PatientID" not in df.columns:
        raise ValueError("geometry table must contain PatientID")

    df["PatientID"] = df["PatientID"].map(norm_id)
    return df.copy()


def base_feature_cols(base):
    exclude = {
        "PatientID",
        "patient_id",
        "case_id",
        "id",
        "ID",
        "T_stage",
        "N_stage",
        "M_stage",
        "target",
        "true",
        "pred",
        "ct_path",
        "tumor_mask_path",
        "hu_bone_status",
    }

    cols = []
    for c in base.columns:
        cl = c.lower()

        if c in exclude:
            continue
        if "path" in cl:
            continue
        if "stage" in cl:
            continue
        if cl in ["target", "true", "pred"]:
            continue

        cols.append(c)

    return cols


def prepare_branch(base, hu_path, geom_path):
    hu = load_hu(hu_path)
    geom = load_geom(geom_path)

    df = base.merge(hu, on="PatientID", how="left")
    df = df.merge(geom, on="PatientID", how="left")

    for c in HU_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    for c in df.columns:
        if c.startswith("fast_geom_") and c != "fast_geom_status":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    bcols = base_feature_cols(base)
    hcols = [c for c in HU_COLS if c in df.columns]
    gcols = [
        c for c in df.columns
        if c.startswith("fast_geom_") and c != "fast_geom_status"
    ]

    feature_cols = []
    seen = set()

    for c in bcols + hcols + gcols:
        if c in df.columns and c not in seen:
            feature_cols.append(c)
            seen.add(c)

    return df, feature_cols


def build_preprocessor(X):
    X = X.copy()

    num_cols = []
    cat_cols = []

    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            num_cols.append(c)
        else:
            converted = pd.to_numeric(X[c], errors="coerce")
            if converted.notna().mean() > 0.8:
                X[c] = converted
                num_cols.append(c)
            else:
                cat_cols.append(c)

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_ohe()),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    return X, pre, num_cols, cat_cols


def train_branch(df, feature_cols, branch_name):
    X_raw = df[feature_cols].copy()
    X_raw, pre, num_cols, cat_cols = build_preprocessor(X_raw)

    y_int = df["T_stage"].map(STAGE_TO_INT).astype(int)

    X_mat = pre.fit_transform(X_raw)

    classifiers = {}

    for thr in [2, 3, 4]:
        y_bin = (y_int >= thr).astype(int)

        clf = RandomForestClassifier(
            n_estimators=450,
            random_state=42,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
        )

        print(
            "[TRAIN]",
            branch_name,
            "T>=" + str(thr),
            "n=" + str(len(df)),
            "features=" + str(len(feature_cols)),
        )

        clf.fit(X_mat, y_bin)
        classifiers["T_ge_" + str(thr)] = clf

    return {
        "branch_name": branch_name,
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "preprocessor": pre,
        "classifiers": classifiers,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_features", required=True)
    ap.add_argument("--hu_petaware", required=True)
    ap.add_argument("--geom_petaware", required=True)
    ap.add_argument("--hu_final", required=True)
    ap.add_argument("--geom_final", required=True)
    ap.add_argument(
        "--out",
        default="tstage_ordinal_best/models/tstage_ordinal_ensemble_final.joblib",
    )
    args = ap.parse_args()

    base = load_base(args.base_features)

    pet_df, pet_cols = prepare_branch(
        base,
        args.hu_petaware,
        args.geom_petaware,
    )

    fin_df, fin_cols = prepare_branch(
        base,
        args.hu_final,
        args.geom_final,
    )

    artifact = {
        "model_name": "tstage_ordinal_rf_probability_ensemble",
        "labels": LABELS,
        "stage_to_int": STAGE_TO_INT,
        "weights": {
            "petaware": 0.80,
            "final": 0.20,
        },
        "thresholds": {
            "T_ge_2": 0.64,
            "T_ge_3": 0.44,
            "T_ge_4": 0.35,
        },
        "branches": {
            "petaware": train_branch(pet_df, pet_cols, "petaware"),
            "final": train_branch(fin_df, fin_cols, "final"),
        },
        "training_info": {
            "base_features": args.base_features,
            "hu_petaware": args.hu_petaware,
            "geom_petaware": args.geom_petaware,
            "hu_final": args.hu_final,
            "geom_final": args.geom_final,
            "n_train": int(len(base)),
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(artifact, args.out, compress=3)

    meta = {
        "model_name": artifact["model_name"],
        "labels": artifact["labels"],
        "weights": artifact["weights"],
        "thresholds": artifact["thresholds"],
        "training_info": artifact["training_info"],
        "n_features_petaware": len(pet_cols),
        "n_features_final": len(fin_cols),
    }

    meta_path = args.out.replace(".joblib", ".metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("saved:", args.out)
    print("saved:", meta_path)


if __name__ == "__main__":
    main()
