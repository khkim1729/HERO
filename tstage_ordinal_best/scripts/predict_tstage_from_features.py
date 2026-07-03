import argparse
import joblib
import numpy as np
import pandas as pd


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


def read_base(path):
    df = pd.read_csv(path)
    if "PatientID" not in df.columns:
        raise ValueError("base_features must contain PatientID")
    df["PatientID"] = df["PatientID"].map(norm_id)
    return df


def read_hu(path):
    df = ensure_hu_features(pd.read_csv(path))

    if "PatientID" not in df.columns:
        if "case_id" not in df.columns:
            raise ValueError("HU table must contain PatientID or case_id")
        df["PatientID"] = df["case_id"].map(norm_id)
    else:
        df["PatientID"] = df["PatientID"].map(norm_id)

    keep = ["PatientID"] + [c for c in HU_COLS if c in df.columns]
    return df[keep].copy()


def read_geom(path):
    df = pd.read_csv(path)
    if "PatientID" not in df.columns:
        raise ValueError("geometry table must contain PatientID")
    df["PatientID"] = df["PatientID"].map(norm_id)
    return df.copy()


def prepare_branch_df(base, hu_path, geom_path, branch):
    hu = read_hu(hu_path)
    geom = read_geom(geom_path)

    df = base.merge(hu, on="PatientID", how="left")
    df = df.merge(geom, on="PatientID", how="left")

    for c in HU_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    for c in df.columns:
        if c.startswith("fast_geom_") and c != "fast_geom_status":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for c in branch["feature_cols"]:
        if c not in df.columns:
            df[c] = 0

    for c in branch.get("num_cols", []):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def predict_branch(branch, df):
    X = df[branch["feature_cols"]].copy()
    X_mat = branch["preprocessor"].transform(X)

    p2 = branch["classifiers"]["T_ge_2"].predict_proba(X_mat)[:, 1]
    p3 = branch["classifiers"]["T_ge_3"].predict_proba(X_mat)[:, 1]
    p4 = branch["classifiers"]["T_ge_4"].predict_proba(X_mat)[:, 1]

    return p2, p3, p4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base_features", required=True)
    ap.add_argument("--hu_petaware", required=True)
    ap.add_argument("--geom_petaware", required=True)
    ap.add_argument("--hu_final", required=True)
    ap.add_argument("--geom_final", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    artifact = joblib.load(args.model)
    base = read_base(args.base_features)

    pet_df = prepare_branch_df(
        base,
        args.hu_petaware,
        args.geom_petaware,
        artifact["branches"]["petaware"],
    )

    fin_df = prepare_branch_df(
        base,
        args.hu_final,
        args.geom_final,
        artifact["branches"]["final"],
    )

    p2_pet, p3_pet, p4_pet = predict_branch(
        artifact["branches"]["petaware"],
        pet_df,
    )

    p2_fin, p3_fin, p4_fin = predict_branch(
        artifact["branches"]["final"],
        fin_df,
    )

    w_pet = artifact["weights"]["petaware"]
    w_fin = artifact["weights"]["final"]

    p2 = w_pet * p2_pet + w_fin * p2_fin
    p3 = w_pet * p3_pet + w_fin * p3_fin
    p4 = w_pet * p4_pet + w_fin * p4_fin

    thr2 = artifact["thresholds"]["T_ge_2"]
    thr3 = artifact["thresholds"]["T_ge_3"]
    thr4 = artifact["thresholds"]["T_ge_4"]

    pred = np.array(["T1"] * len(base), dtype=object)
    pred[p2 >= thr2] = "T2"
    pred[p3 >= thr3] = "T3"
    pred[p4 >= thr4] = "T4"

    out = pd.DataFrame(
        {
            "PatientID": base["PatientID"].values,
            "p_ge_T2": p2,
            "p_ge_T3": p3,
            "p_ge_T4": p4,
            "pred_T_stage": pred,
        }
    )

    out.to_csv(args.out, index=False)
    print("saved:", args.out)
    print(out.head().to_string(index=False))


if __name__ == "__main__":
    main()
