from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.transforms import EnsureTyped
from torch import nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hero.data.dataset import build_case_records, split_folds
from hero.data.transforms import get_infer_transforms, get_train_transforms
from hero.utils.config import load_yaml, resolve_path
from hero.utils.metrics import dice_coefficient
from hero.utils.training import ensure_dir, save_json, set_seed, to_device


def choose_data_root(common_cfg: dict) -> Path:
    project_root = Path(common_cfg["paths"]["project_root"]).resolve()
    full_root = resolve_path(project_root, common_cfg["paths"]["full_training_root"])
    sample_root = resolve_path(project_root, common_cfg["paths"]["data_root"])
    return full_root if full_root.exists() else sample_root


def build_model(model_cfg: dict) -> nn.Module:
    return UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        channels=tuple(model_cfg["channels"]),
        strides=tuple(model_cfg["strides"]),
        num_res_units=model_cfg["num_res_units"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: OOF label cleaning for HECKTOR 2026.")
    parser.add_argument("--common-config", default="configs/common_config.yaml")
    parser.add_argument("--step-config", default="configs/step1_label_cleaning.yaml")
    args = parser.parse_args()

    common_cfg = load_yaml(args.common_config)
    step_cfg = load_yaml(args.step_config)
    model_cfg = step_cfg["model"]
    set_seed(common_cfg["seed"], common_cfg["runtime"]["deterministic"])

    data_root = choose_data_root(common_cfg)
    project_root = Path(common_cfg["paths"]["project_root"]).resolve()
    clinical_csv = resolve_path(project_root, common_cfg["paths"]["clinical_csv"])
    output_index = resolve_path(project_root, common_cfg["paths"]["cleaned_index"])
    ensure_dir(output_index.parent)

    records, _ = build_case_records(data_root, clinical_csv)
    folds = split_folds(records, step_cfg["data"]["folds"], common_cfg["seed"])
    pixdim = tuple(step_cfg["data"]["pixdim"])
    roi_size = tuple(step_cfg["data"]["roi_size"])
    train_transforms = get_train_transforms(pixdim=pixdim, roi_size=roi_size)
    eval_transforms = get_infer_transforms(pixdim=pixdim, roi_size=roi_size)

    device_name = common_cfg["runtime"]["device"]
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=step_cfg["inference"]["sw_batch_size"],
        overlap=step_cfg["inference"]["overlap"],
    )
    loss_fn = DiceCELoss(sigmoid=True)

    per_case_diff = {}
    per_case_dice = {}

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_ds = Dataset(data=[records[i] for i in train_idx], transform=train_transforms)
        val_ds = Dataset(data=[records[i] for i in val_idx], transform=eval_transforms)
        train_loader = DataLoader(
            train_ds,
            batch_size=step_cfg["data"]["batch_size"],
            shuffle=True,
            num_workers=common_cfg["runtime"]["num_workers"],
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=step_cfg["data"]["val_batch_size"],
            shuffle=False,
            num_workers=common_cfg["runtime"]["num_workers"],
        )

        model = build_model(model_cfg).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=step_cfg["optimization"]["learning_rate"],
            weight_decay=step_cfg["optimization"]["weight_decay"],
        )
        scaler = torch.amp.GradScaler(enabled=bool(common_cfg["runtime"]["amp"] and device.type == "cuda"))

        model.train()
        for _ in range(step_cfg["optimization"]["epochs"]):
            for batch in train_loader:
                batch = to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    logits = model(batch["image"])
                    loss = loss_fn(logits, batch["label"])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Fold {fold_idx + 1}/{len(folds)}"):
                batch = to_device(batch, device)
                logits = inferer(inputs=batch["image"], network=model)
                probs = torch.sigmoid(logits)
                labels = batch["label"]
                patient_ids = batch["patient_id"]
                for idx, patient_id in enumerate(patient_ids):
                    diff = torch.abs((probs[idx] > 0.5).float() - labels[idx]).mean().item()
                    dice = dice_coefficient(probs[idx], labels[idx])
                    per_case_diff[patient_id] = diff
                    per_case_dice[patient_id] = dice

    diffs = np.asarray(list(per_case_diff.values()), dtype=float)
    threshold = float(np.percentile(diffs, step_cfg["label_cleaning"]["noisy_diff_percentile"])) if len(diffs) else 1.0
    clean_patient_ids = sorted([pid for pid, diff in per_case_diff.items() if diff <= threshold])
    noisy_patient_ids = sorted([pid for pid, diff in per_case_diff.items() if diff > threshold])
    payload = {
        "data_root": str(data_root),
        "clinical_csv": str(clinical_csv),
        "noisy_diff_threshold": threshold,
        "clean_patient_ids": clean_patient_ids,
        "noisy_patient_ids": noisy_patient_ids,
        "per_case_diff": per_case_diff,
        "per_case_dice": per_case_dice,
    }
    save_json(payload, output_index)
    print(f"Saved cleaned dataset index to {output_index}")
    print(f"Retained {len(clean_patient_ids)} / {len(records)} patients.")


if __name__ == "__main__":
    main()
