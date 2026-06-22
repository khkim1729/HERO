from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def _norm(volume: np.ndarray) -> np.ndarray:
    low, high = np.percentile(volume, [1, 99])
    volume = np.clip(volume, low, high)
    return (volume - volume.min()) / max(volume.max() - volume.min(), 1e-6)


def _pick_case(data_root: Path) -> tuple[str, Path, Path, Path]:
    for patient_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        patient_id = patient_dir.name
        label_path = patient_dir / f"{patient_id}.nii.gz"
        ct_path = patient_dir / f"{patient_id}__CT.nii.gz"
        pt_path = patient_dir / f"{patient_id}__PT.nii.gz"
        if label_path.exists() and ct_path.exists() and pt_path.exists():
            return patient_id, ct_path, pt_path, label_path
    raise FileNotFoundError(f"No complete patient folders found in {data_root}")


def _target_slices(volume: np.ndarray, center: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axial = volume[:, :, center[2]]
    coronal = volume[:, center[1], :]
    sagittal = volume[center[0], :, :]
    return axial, coronal, sagittal


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PET/CT/label slice grid for HECKTOR sample data.")
    parser.add_argument("--data-root", default="data/sample_5")
    parser.add_argument("--output", default="sample_slice_grid.png")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    patient_id, ct_path, pt_path, label_path = _pick_case(data_root)
    ct = _norm(nib.load(str(ct_path)).get_fdata().astype(np.float32))
    pt = _norm(nib.load(str(pt_path)).get_fdata().astype(np.float32))
    label = nib.load(str(label_path)).get_fdata().astype(np.float32)
    coords = np.argwhere(label > 0)
    if len(coords) > 0:
        center = tuple(np.round(coords.mean(axis=0)).astype(int).tolist())
    else:
        center = (label.shape[0] // 2, label.shape[1] // 2, label.shape[2] // 2)

    volumes = {
        "CT": _target_slices(ct, center),
        "PET": _target_slices(pt, center),
        "Label": _target_slices(label, center),
    }
    planes = ["Axial", "Coronal", "Sagittal"]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f"HECKTOR 2026 Sample Views: {patient_id}", fontsize=16)
    for row_idx, (modality, slices) in enumerate(volumes.items()):
        for col_idx, slice_2d in enumerate(slices):
            ax = axes[row_idx, col_idx]
            ax.imshow(np.rot90(slice_2d), cmap="gray")
            if modality != "Label":
                ax.contour(np.rot90(volumes["Label"][col_idx]), levels=[0.5], colors="lime", linewidths=0.8)
            ax.set_title(f"{modality} {planes[col_idx]}")
            ax.axis("off")

    fig.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved slice grid to {args.output}")


if __name__ == "__main__":
    main()
