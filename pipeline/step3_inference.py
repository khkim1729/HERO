from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import pandas as pd
import torch
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hero.data.dataset import build_case_records
from hero.data.transforms import get_infer_transforms
from hero.models.multitask_uxnet import MultiTaskUXNet
from hero.utils.config import load_yaml, resolve_path
from hero.utils.training import ensure_dir, set_seed, to_device


def center_crop_or_pad(image: torch.Tensor, roi_size: tuple[int, int, int]) -> torch.Tensor:
    _, _, h, w, d = image.shape
    target_h, target_w, target_d = roi_size
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    pad_d = max(target_d - d, 0)
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        image = F.pad(
            image,
            (
                pad_d // 2,
                pad_d - pad_d // 2,
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
            ),
        )
        _, _, h, w, d = image.shape

    start_h = max((h - target_h) // 2, 0)
    start_w = max((w - target_w) // 2, 0)
    start_d = max((d - target_d) // 2, 0)
    return image[:, :, start_h:start_h + target_h, start_w:start_w + target_w, start_d:start_d + target_d]


def choose_data_root(common_cfg: dict) -> Path:
    project_root = Path(common_cfg["paths"]["project_root"]).resolve()
    full_root = resolve_path(project_root, common_cfg["paths"]["full_training_root"])
    sample_root = resolve_path(project_root, common_cfg["paths"]["data_root"])
    return full_root if full_root.exists() else sample_root


def pick_checkpoint(checkpoint_dir: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    candidates = sorted(checkpoint_dir.glob("best_model_epoch_*_dice_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No multitask checkpoint found under {checkpoint_dir}")
    return candidates[-1]


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[MultiTaskUXNet, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint["model_config"]
    model = MultiTaskUXNet(
        in_channels=model_cfg["in_channels"],
        encoder_dims=model_cfg["encoder_dims"],
        depths=model_cfg["depths"],
        kernel_size=model_cfg["kernel_size"],
        dropout=model_cfg["dropout"],
        seg_out_channels=model_cfg["seg_out_channels"],
        t_stage_classes=model_cfg["t_stage_classes"],
        n_stage_classes=model_cfg["n_stage_classes"],
        clinical_feature_dim=checkpoint["clinical_feature_dim"],
        staging_feature_dim=checkpoint["staging_feature_dim"],
        clinical_embed_dim=model_cfg["clinical_embed_dim"],
        mlp_hidden_dim=model_cfg["mlp_hidden_dim"],
        prognosis_hidden_dim=model_cfg["prognosis_hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: T4-optimized multitask inference.")
    parser.add_argument("--common-config", default="configs/common_config.yaml")
    parser.add_argument("--step-config", default="configs/step3_inference.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    common_cfg = load_yaml(args.common_config)
    step_cfg = load_yaml(args.step_config)
    set_seed(common_cfg["seed"], common_cfg["runtime"]["deterministic"])
    project_root = Path(common_cfg["paths"]["project_root"]).resolve()
    data_root = choose_data_root(common_cfg)
    clinical_csv = resolve_path(project_root, common_cfg["paths"]["clinical_csv"])
    cleaned_index = resolve_path(project_root, common_cfg["paths"]["cleaned_index"])
    checkpoint_dir = resolve_path(project_root, common_cfg["paths"]["checkpoints_root"])
    checkpoint_path = pick_checkpoint(checkpoint_dir, args.checkpoint)

    device_name = common_cfg["runtime"]["device"]
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    model, _ = load_model(checkpoint_path, device)
    records, _ = build_case_records(data_root, clinical_csv, cleaned_index_path=cleaned_index)
    pixdim = tuple(step_cfg["data"]["pixdim"])
    roi_size = tuple(step_cfg["data"]["roi_size"])

    loader = DataLoader(
        Dataset(records, transform=get_infer_transforms(pixdim=pixdim, roi_size=roi_size)),
        batch_size=1,
        shuffle=False,
        num_workers=common_cfg["runtime"]["num_workers"],
    )
    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=step_cfg["inference"]["sw_batch_size"],
        overlap=step_cfg["inference"]["overlap"],
    )

    out_dir = ensure_dir(project_root / "outputs" / "inference")
    seg_dir = ensure_dir(out_dir / "segmentation")
    table_rows = []

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            patient_id = batch["patient_id"][0]
            outputs = inferer(
                inputs=batch["image"],
                network=lambda x: model(x, batch["staging_features"].float(), batch["clinical_features"].float())["seg_logits"],
            )
            cls_image = center_crop_or_pad(batch["image"], roi_size)
            logits = model(
                cls_image,
                batch["staging_features"].float(),
                batch["clinical_features"].float(),
            )
            seg_prob = torch.sigmoid(outputs)[0, 0]
            pred_mask = (seg_prob > step_cfg["inference"]["threshold"]).float()

            ct_path = Path(batch["ct_path"][0])
            original = nib.load(str(ct_path))
            restored = F.interpolate(
                pred_mask.unsqueeze(0).unsqueeze(0),
                size=original.shape,
                mode="nearest",
            )[0, 0].cpu().numpy().astype("uint8")
            nib.save(nib.Nifti1Image(restored, original.affine, original.header), seg_dir / f"{patient_id}.nii.gz")

            t_pred = int(torch.argmax(logits["t_logits"], dim=1).item())
            n_pred = int(torch.argmax(logits["n_logits"], dim=1).item())
            relapse_score = float(torch.sigmoid(logits["relapse_logit"]).item())
            rfs_risk = float(logits["rfs_logit"].item())
            table_rows.append({
                "PatientID": patient_id,
                "T_stage_pred": t_pred,
                "N_stage_pred": n_pred,
                "Relapse_score": relapse_score,
                "RFS_risk": rfs_risk,
                "SegmentationPath": str(seg_dir / f"{patient_id}.nii.gz"),
            })

    pred_csv = out_dir / "staging_prognosis_predictions.csv"
    pd.DataFrame(table_rows).to_csv(pred_csv, index=False)
    print(f"Saved segmentation masks to {seg_dir}")
    print(f"Saved task 2/3 predictions to {pred_csv}")


if __name__ == "__main__":
    main()
