from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = torch.sum(pred * target).item()
    denom = torch.sum(pred).item() + torch.sum(target).item()
    return float((2.0 * intersection + eps) / (denom + eps))


def binary_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) > 0.5).long()
    return float((pred == target.long()).float().mean().item())


def multiclass_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == target.long()).float().mean().item())


def masked_multiclass_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    mask = target >= 0
    if mask.sum() == 0:
        return 0.0
    pred = logits.argmax(dim=1)
    return float((pred[mask] == target[mask].long()).float().mean().item())


def concordance_index(risk_scores: Iterable[float], times: Iterable[float], events: Iterable[float]) -> float:
    risks = np.asarray(list(risk_scores), dtype=float)
    times = np.asarray(list(times), dtype=float)
    events = np.asarray(list(events), dtype=float)
    comparable = 0
    concordant = 0.0

    for i in range(len(risks)):
        for j in range(i + 1, len(risks)):
            if times[i] == times[j]:
                continue
            if events[i] == 0 and events[j] == 0:
                continue

            if times[i] < times[j] and events[i] == 1:
                comparable += 1
                if risks[i] > risks[j]:
                    concordant += 1
                elif risks[i] == risks[j]:
                    concordant += 0.5
            elif times[j] < times[i] and events[j] == 1:
                comparable += 1
                if risks[j] > risks[i]:
                    concordant += 1
                elif risks[i] == risks[j]:
                    concordant += 0.5

    if comparable == 0:
        return 0.5
    return float(concordant / comparable)
