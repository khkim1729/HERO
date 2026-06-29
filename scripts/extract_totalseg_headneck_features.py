from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


HEADNECK_KEYWORDS = [
    "skull",
    "mandible",
    "clavicula",
    "vertebrae",
    "rib",
    "hyoid",
    "sternum",
    "scapula",
    "carotid",
    "jugular",
    "aorta",
    "brachiocephalic",
    "subclavian",
    "thyroid",
    "trachea",
    "esophagus",
    "spinal",
]


def load_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(str(path))
    data = img.get_fdata()
    mask = data > 0
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return mask, zooms


def voxel_volume_mm3(zooms: tuple[float, float, float]) -> float:
    return float(zooms[0] * zooms[1] * zooms[2])


def center_of_mass(mask: np.ndarray, zooms: tuple[float, float, float]) -> tuple[float, float, float] | None:
    if mask.sum() == 0:
        return None
    c = ndimage.center_of_mass(mask.astype(np.uint8))
    return tuple(float(c[i] * zooms[i]) for i in range(3))


def min_surface_distance_mm(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    zooms: tuple[float, float, float],
) -> float:
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return float("nan")

    inv_b = ~mask_b
    dist_to_b = ndimage.distance_transform_edt(inv_b, sampling=zooms)
    return float(dist_to_b[mask_a].min())


def find_totalseg_masks(totalseg_dir: Path) -> list[Path]:
    nii_files = list(totalseg_dir.glob("*.nii.gz")) + list(totalseg_dir.glob("*.nii"))

    selected = []
    for p in nii_files:
        name = p.name.lower()
        if any(k in name for k in HEADNECK_KEYWORDS):
            selected.append(p)

    return sorted(selected)


def get_feature_columns_from_model(model_path: Path) -> list[str]:
    package = joblib.load(model_path)
    metrics = package.get("metrics", {})
    cols = metrics.get("feature_columns")
    if cols is None:
        cols = package.get("feature_columns")
    if cols is None:
        raise RuntimeError(f"Could not find feature_columns in model package: {model_path}")
    return list(cols)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract simple TotalSegmentator anatomy features for RF TN staging inference."
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--ct", required=True)
    parser.add_argument("--pet", required=True)
    parser.add_argument("--tumor-mask", required=True, help="Primary tumor or GTVp mask.")
    parser.add_argument("--node-mask", required=True, help="Nodal or GTVn mask.")
    parser.add_argument("--totalseg-dir", required=True)
    parser.add_argument("--rf-model", required=True, help="RF model .joblib used to obtain feature schema.")
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    case_id = args.case_id
    ct_path = Path(args.ct)
    pet_path = Path(args.pet)
    tumor_mask_path = Path(args.tumor_mask)
    node_mask_path = Path(args.node_mask)
    totalseg_dir = Path(args.totalseg_dir)
    rf_model_path = Path(args.rf_model)
    out_csv = Path(args.out_csv)

    for p in [ct_path, pet_path, tumor_mask_path, node_mask_path, totalseg_dir, rf_model_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required path not found: {p}")

    tumor_mask, zooms = load_mask(tumor_mask_path)
    node_mask, _ = load_mask(node_mask_path)
    pet_img = nib.load(str(pet_path))
    pet = pet_img.get_fdata()

    vv = voxel_volume_mm3(zooms)

    raw_features: dict[str, Any] = {
        "PatientID": case_id,
        "case_id": case_id,
        "gtvp_volume_mm3": float(tumor_mask.sum() * vv),
        "gtvn_volume_mm3": float(node_mask.sum() * vv),
        "gtvp_volume_ml": float(tumor_mask.sum() * vv / 1000.0),
        "gtvn_volume_ml": float(node_mask.sum() * vv / 1000.0),
        "gtvn_component_count": int(ndimage.label(node_mask)[1]),
        "gtvp_pet_mean": float(np.nanmean(pet[tumor_mask])) if tumor_mask.sum() else float("nan"),
        "gtvp_pet_max": float(np.nanmax(pet[tumor_mask])) if tumor_mask.sum() else float("nan"),
        "gtvn_pet_mean": float(np.nanmean(pet[node_mask])) if node_mask.sum() else float("nan"),
        "gtvn_pet_max": float(np.nanmax(pet[node_mask])) if node_mask.sum() else float("nan"),
    }

    tumor_com = center_of_mass(tumor_mask, zooms)
    node_com = center_of_mass(node_mask, zooms)
    if tumor_com is not None and node_com is not None:
        raw_features["gtvp_gtvn_com_distance_mm"] = float(np.linalg.norm(np.array(tumor_com) - np.array(node_com)))
    else:
        raw_features["gtvp_gtvn_com_distance_mm"] = float("nan")

    for mask_path in find_totalseg_masks(totalseg_dir):
        anatomy_name = mask_path.name.replace(".nii.gz", "").replace(".nii", "")
        anatomy_mask, _ = load_mask(mask_path)

        prefix = f"totalseg_{anatomy_name}"
        raw_features[f"{prefix}_volume_mm3"] = float(anatomy_mask.sum() * vv)
        raw_features[f"{prefix}_gtvp_overlap_mm3"] = float((anatomy_mask & tumor_mask).sum() * vv)
        raw_features[f"{prefix}_gtvn_overlap_mm3"] = float((anatomy_mask & node_mask).sum() * vv)
        raw_features[f"{prefix}_gtvp_min_distance_mm"] = min_surface_distance_mm(tumor_mask, anatomy_mask, zooms)
        raw_features[f"{prefix}_gtvn_min_distance_mm"] = min_surface_distance_mm(node_mask, anatomy_mask, zooms)

    feature_columns = get_feature_columns_from_model(rf_model_path)

    row = {}
    for col in feature_columns:
        row[col] = raw_features.get(col, np.nan)

    row["PatientID"] = case_id
    row["case_id"] = case_id

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_csv, index=False)

    schema_path = out_csv.with_suffix(".feature_schema.json")
    schema_path.write_text(
        json.dumps(
            {
                "case_id": case_id,
                "rf_model": str(rf_model_path),
                "num_required_feature_columns": len(feature_columns),
                "num_available_raw_features": len(raw_features),
                "num_missing_model_features_filled_with_nan": int(sum(pd.isna(row[c]) for c in feature_columns)),
                "feature_columns": feature_columns,
                "available_raw_feature_names": sorted(raw_features.keys()),
            },
            indent=2,
        )
    )

    print(f"Wrote feature CSV: {out_csv}")
    print(f"Wrote schema report: {schema_path}")


if __name__ == "__main__":
    main()
