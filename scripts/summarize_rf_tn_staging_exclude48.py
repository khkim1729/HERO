from pathlib import Path
import json
import pandas as pd

ROOT = Path("results/tn_staging_rf_exclude48")

rows = []

for summary_path in ROOT.glob("*/*_summary.json"):
    summary = json.loads(summary_path.read_text())
    run_name = summary["run_name"]

    for target in ["T", "N"]:
        m = summary.get("metrics", {}).get(target)
        if not m:
            continue

        rows.append(
            {
                "run_name": run_name,
                "target": target,
                "num_rows_before_exclusion": summary["num_rows_before_exclusion"],
                "num_rows_after_exclusion": summary["num_rows_after_exclusion"],
                "num_removed_rows": summary["num_removed_rows"],
                "num_features": m["num_features"],
                "n_splits": m["n_splits"],
                "oof_accuracy": m["oof_accuracy"],
                "oof_balanced_accuracy": m["oof_balanced_accuracy"],
                "oof_macro_f1": m["oof_macro_f1"],
            }
        )

out = ROOT / "rf_exclude48_summary_table.csv"
df = pd.DataFrame(rows)
df.to_csv(out, index=False)

print(df.sort_values(["target", "oof_macro_f1"], ascending=[True, False]))
print("")
print(f"Saved: {out}")
