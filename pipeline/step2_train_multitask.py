from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from monai.losses import DiceCELoss
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

from hero.data.dataset import build_case_records
from hero.data.transforms import get_eval_transforms, get_train_transforms
from hero.models.multitask_uxnet import MultiTaskUXNet
from hero.utils.config import load_yaml, resolve_path
from hero.utils.metrics import binary_accuracy, concordance_index, dice_coefficient, masked_multiclass_accuracy
from hero.utils.training import ensure_dir, save_json, set_seed, to_device


def choose_data_root(common_cfg: dict) -> Path:
    project_root = Path(common_cfg["paths"]["project_root"]).resolve()
    full_root = resolve_path(project_root, common_cfg["paths"]["full_training_root"])
    sample_root = resolve_path(project_root, common_cfg["paths"]["data_root"])
    return full_root if full_root.exists() else sample_root


def masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = target >= 0
    if mask.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], target[mask])


def build_model(model_cfg: dict, clinical_dim: int, staging_dim: int) -> nn.Module:
    return MultiTaskUXNet(
        in_channels=model_cfg["in_channels"],
        encoder_dims=model_cfg["encoder_dims"],
        depths=model_cfg["depths"],
        kernel_size=model_cfg["kernel_size"],
        dropout=model_cfg["dropout"],
        seg_out_channels=model_cfg["seg_out_channels"],
        t_stage_classes=model_cfg["t_stage_classes"],
        n_stage_classes=model_cfg["n_stage_classes"],
        clinical_feature_dim=clinical_dim,
        staging_feature_dim=staging_dim,
        clinical_embed_dim=model_cfg["clinical_embed_dim"],
        mlp_hidden_dim=model_cfg["mlp_hidden_dim"],
        prognosis_hidden_dim=model_cfg["prognosis_hidden_dim"],
    )


def compute_losses(outputs: dict[str, torch.Tensor], batch: dict, loss_cfg: dict, seg_loss: nn.Module) -> tuple[torch.Tensor, dict[str, float]]:
    seg = seg_loss(outputs["seg_logits"], batch["label"]) * loss_cfg["seg_weight"]
    t_loss = masked_cross_entropy(outputs["t_logits"], batch["t_stage"]) * loss_cfg["t_stage_weight"]
    n_loss = masked_cross_entropy(outputs["n_logits"], batch["n_stage"]) * loss_cfg["n_stage_weight"]
    relapse = F.binary_cross_entropy_with_logits(
        outputs["relapse_logit"].squeeze(1),
        batch["relapse"].float(),
    ) * loss_cfg["relapse_weight"]
    rfs_target = -torch.log1p(batch["rfs_time"])
    rfs = F.mse_loss(outputs["rfs_logit"].squeeze(1), rfs_target) * loss_cfg["rfs_weight"]
    total = seg + t_loss + n_loss + relapse + rfs
    return total, {
        "seg": float(seg.item()),
        "t": float(t_loss.item()),
        "n": float(n_loss.item()),
        "relapse": float(relapse.item()),
        "rfs": float(rfs.item()),
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    dice_scores = []
    t_acc = []
    n_acc = []
    relapse_acc = []
    risks = []
    times = []
    events = []

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            outputs = model(
                batch["image"],
                batch["staging_features"].float(),
                batch["clinical_features"].float(),
            )
            probs = torch.sigmoid(outputs["seg_logits"])
            dice_scores.extend([dice_coefficient(probs[idx], batch["label"][idx]) for idx in range(probs.shape[0])])
            t_acc.append(masked_multiclass_accuracy(outputs["t_logits"], batch["t_stage"]))
            n_acc.append(masked_multiclass_accuracy(outputs["n_logits"], batch["n_stage"]))
            relapse_acc.append(binary_accuracy(outputs["relapse_logit"].squeeze(1), batch["relapse"]))
            risks.extend(outputs["rfs_logit"].squeeze(1).detach().cpu().tolist())
            times.extend(batch["rfs_time"].detach().cpu().tolist())
            events.extend(batch["rfs_event"].detach().cpu().tolist())

    return {
        "dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "t_acc": float(np.mean(t_acc)) if t_acc else 0.0,
        "n_acc": float(np.mean(n_acc)) if n_acc else 0.0,
        "relapse_acc": float(np.mean(relapse_acc)) if relapse_acc else 0.0,
        "cindex": concordance_index(risks, times, events),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: Train the HECKTOR 2026 multi-task UX-Net.")
    parser.add_argument("--common-config", default="configs/common_config.yaml")
    parser.add_argument("--model-config", default="configs/model_configs.yaml")
    args = parser.parse_args()

    common_cfg = load_yaml(args.common_config)
    model_bundle = load_yaml(args.model_config)
    model_cfg = model_bundle["multitask_model"]
    loss_cfg = model_bundle["losses"]
    set_seed(common_cfg["seed"], common_cfg["runtime"]["deterministic"])

    project_root = Path(common_cfg["paths"]["project_root"]).resolve()
    data_root = choose_data_root(common_cfg)
    clinical_csv = resolve_path(project_root, common_cfg["paths"]["clinical_csv"])
    cleaned_index = resolve_path(project_root, common_cfg["paths"]["cleaned_index"])
    ckpt_dir = ensure_dir(resolve_path(project_root, common_cfg["paths"]["checkpoints_root"]))

    records, processor = build_case_records(data_root, clinical_csv, cleaned_index_path=cleaned_index)
    train_records, val_records = train_test_split(records, test_size=0.2, random_state=common_cfg["seed"])
    train_loader = DataLoader(
        Dataset(train_records, transform=get_train_transforms()),
        batch_size=common_cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=common_cfg["runtime"]["num_workers"],
    )
    val_loader = DataLoader(
        Dataset(val_records, transform=get_eval_transforms()),
        batch_size=common_cfg["data"]["val_batch_size"],
        shuffle=False,
        num_workers=common_cfg["runtime"]["num_workers"],
    )

    sample_record = records[0]
    clinical_dim = len(sample_record["clinical_features"])
    staging_dim = len(sample_record["staging_features"])

    device_name = common_cfg["runtime"]["device"]
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    model = build_model(model_cfg, clinical_dim, staging_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=common_cfg["optimization"]["learning_rate"],
        weight_decay=common_cfg["optimization"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler(enabled=bool(common_cfg["runtime"]["amp"] and device.type == "cuda"))
    seg_loss = DiceCELoss(sigmoid=True)

    best_score = -math.inf
    history = []
    saved_paths = []

    for epoch in range(1, common_cfg["optimization"]["max_epochs"] + 1):
        model.train()
        epoch_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                outputs = model(
                    batch["image"],
                    batch["staging_features"].float(),
                    batch["clinical_features"].float(),
                )
                loss, _ = compute_losses(outputs, batch, loss_cfg, seg_loss)
            scaler.scale(loss).backward()
            if common_cfg["optimization"]["grad_clip_norm"] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), common_cfg["optimization"]["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.item()))

        metrics = evaluate(model, val_loader, device)
        metrics["loss"] = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        metrics["epoch"] = epoch
        history.append(metrics)

        score = metrics["dice"] + metrics["cindex"]
        checkpoint_path = ckpt_dir / (
            f"best_model_epoch_{epoch}_dice_{metrics['dice']:.4f}_cindex_{metrics['cindex']:.4f}.pth"
        )
        if score > best_score:
            best_score = score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "clinical_feature_dim": clinical_dim,
                "staging_feature_dim": staging_dim,
                "clinical_feature_names": processor.feature_names_,
                "model_config": model_cfg,
            }, checkpoint_path)
            saved_paths.append(str(checkpoint_path))

        print(
            f"Epoch {epoch}: loss={metrics['loss']:.4f} dice={metrics['dice']:.4f} "
            f"t_acc={metrics['t_acc']:.4f} n_acc={metrics['n_acc']:.4f} cindex={metrics['cindex']:.4f}"
        )

    save_json({"history": history, "saved_checkpoints": saved_paths}, ckpt_dir / "training_history.json")
    print(f"Training complete. Best composite score: {best_score:.4f}")


if __name__ == "__main__":
    main()
