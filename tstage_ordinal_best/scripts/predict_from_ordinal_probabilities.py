import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--petaware_oof", required=True)
ap.add_argument("--final_oof", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--w_petaware", type=float, default=0.80)
ap.add_argument("--w_final", type=float, default=0.20)
ap.add_argument("--thr_t2", type=float, default=0.64)
ap.add_argument("--thr_t3", type=float, default=0.44)
ap.add_argument("--thr_t4", type=float, default=0.35)
args = ap.parse_args()

pet = pd.read_csv(args.petaware_oof)
fin = pd.read_csv(args.final_oof)

df = pet[["PatientID", "p_ge_T2", "p_ge_T3", "p_ge_T4"]].merge(
    fin[["PatientID", "p_ge_T2", "p_ge_T3", "p_ge_T4"]],
    on="PatientID",
    suffixes=("_petaware", "_final"),
)

df["p_ge_T2"] = args.w_petaware * df["p_ge_T2_petaware"] + args.w_final * df["p_ge_T2_final"]
df["p_ge_T3"] = args.w_petaware * df["p_ge_T3_petaware"] + args.w_final * df["p_ge_T3_final"]
df["p_ge_T4"] = args.w_petaware * df["p_ge_T4_petaware"] + args.w_final * df["p_ge_T4_final"]

pred = np.array(["T1"] * len(df), dtype=object)
pred[df["p_ge_T2"].values >= args.thr_t2] = "T2"
pred[df["p_ge_T3"].values >= args.thr_t3] = "T3"
pred[df["p_ge_T4"].values >= args.thr_t4] = "T4"

df["pred_T_stage"] = pred
df[["PatientID", "p_ge_T2", "p_ge_T3", "p_ge_T4", "pred_T_stage"]].to_csv(args.out, index=False)
print("saved:", args.out)
