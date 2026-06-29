from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TotalSegmentator on input CT for anatomy segmentation."
    )
    parser.add_argument("--ct", required=True, help="Input CT NIfTI path.")
    parser.add_argument("--out-dir", required=True, help="Output directory for TotalSegmentator result.")
    parser.add_argument(
        "--task",
        default="total",
        help="TotalSegmentator task. Default: total",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use --fast option if supported by installed TotalSegmentator.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device argument passed to TotalSegmentator, e.g. gpu or cpu if supported.",
    )
    args = parser.parse_args()

    ct = Path(args.ct)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ct.exists():
        raise FileNotFoundError(f"CT file not found: {ct}")

    cmd = [
        "TotalSegmentator",
        "-i",
        str(ct),
        "-o",
        str(out_dir),
        "--task",
        args.task,
    ]

    if args.fast:
        cmd.append("--fast")

    if args.device is not None:
        cmd.extend(["--device", args.device])

    print("Running command:")
    print(" ".join(cmd), flush=True)

    subprocess.run(cmd, check=True)

    marker = out_dir / "_TOTALSEG_DONE.txt"
    marker.write_text("done\n")
    print(f"TotalSegmentator finished. Output: {out_dir}")


if __name__ == "__main__":
    main()
