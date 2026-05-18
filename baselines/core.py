"""FedAvg / FedProx：本地训练、加权聚合、测试评估。"""

from __future__ import annotations

import copy
import logging
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        correct, total, test_loss = 0, 0, 0.0
        for x, target in test_loader:
            x, target = x.to(device), target.to(device, dtype=torch.int64)
            pred = model(x)
            test_loss += F.cross_entropy(pred, target, reduction="sum").item()
            _, pred_label = torch.max(pred.data, 1)
            total += x.size(0)
            correct += (pred_label == target).sum().item()
    acc = correct / float(total) if total > 0 else 0.0
    return acc, test_loss / float(total) if total > 0 else 0.0


def _proximal_loss(model: nn.Module, global_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    s = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = global_state[name].detach()
        s = s + (p - g).pow(2).sum()
    return 0.5 * s


def local_train_one_client(
    model: nn.Module,
    global_state: Dict[str, torch.Tensor],
    train_loader: DataLoader,
    device: torch.device,
    local_epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    mu: float,
) -> Dict[str, torch.Tensor]:
    """
    在单客户端数据上训练 local_epochs 个 epoch。
    mu>0 时为 FedProx：损失 += (mu/2)||w - w_global||^2；mu=0 为 FedAvg。
    """
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for _ in range(local_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).long()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if mu > 0:
                loss = loss + mu * _proximal_loss(model, global_state)
            loss.backward()
            optimizer.step()

    return copy.deepcopy(model.state_dict())


def fedavg_aggregate(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[int],
) -> Dict[str, torch.Tensor]:
    """按样本数加权平均各客户端 state_dict。"""
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("总样本数为 0")
    keys = state_dicts[0].keys()
    out: Dict[str, torch.Tensor] = {}
    w_norm = [w / total for w in weights]
    for key in keys:
        acc = None
        for sd, w in zip(state_dicts, w_norm):
            t = sd[key].float()
            acc = t * w if acc is None else acc + t * w
        out[key] = acc.to(state_dicts[0][key].dtype)
    return out


def select_client_indices(num_clients: int, join_ratio: float) -> List[int]:
    if join_ratio >= 1.0:
        return list(range(num_clients))
    m = max(1, int(round(num_clients * join_ratio)))
    return sorted(random.sample(range(num_clients), m))
