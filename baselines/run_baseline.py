#!/usr/bin/env python3
"""
FedAvg / FedProx 基线：与主实验共用 Dirichlet 划分 JSON、get_dataset、get_model。

用法（在项目根目录）:
  python -m baselines.run_baseline --algorithm fedavg --dataset CIFAR10 --model ConvNetBN \\
    --alpha 0.05 --client_num 10 --communication_rounds 10 --local_epochs 5 \\
    --lr 0.01 --device cuda:0 --tag baseline1

  python -m baselines.run_baseline --algorithm fedprox --mu 0.01 ...
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from baselines.core import (
    evaluate_model,
    fedavg_aggregate,
    local_train_one_client,
    select_client_indices,
)
from dataset.data.dataset import get_dataset
from src.utils import get_model, setup_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FedAvg / FedProx baselines (same split as main.py)")
    p.add_argument("--algorithm", type=str, choices=["fedavg", "fedprox"], required=True)
    p.add_argument("--mu", type=float, default=0.01, help="FedProx 近端系数（fedavg 时忽略）")
    p.add_argument("--seed", type=int, default=19260817)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dataset_root", type=str, default="./dataset/torchvision")
    p.add_argument("--dataset", type=str, default="CIFAR10")
    p.add_argument("--client_num", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--model", type=str, default="ConvNetBN")
    p.add_argument("--communication_rounds", type=int, default=10)
    p.add_argument("--join_ratio", type=float, default=1.0)
    p.add_argument("--local_epochs", type=int, default=5, help="每轮每客户端本地 epoch 数 E")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01, help="客户端本地 SGD 学习率")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--eval_gap", type=int, default=1, help="每多少通信轮做一次测试")
    p.add_argument("--tag", type=str, default="0")
    p.add_argument(
        "--split_file",
        type=str,
        default="",
        help="可选：显式 JSON；默认 dataset/split_file/{Dataset}_client_num={K}_alpha={alpha}.json",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    mu = args.mu if args.algorithm == "fedprox" else 0.0

    if args.split_file:
        split_path = args.split_file
    else:
        split_path = os.path.join(
            ROOT,
            "dataset",
            "split_file",
            f"{args.dataset}_client_num={args.client_num}_alpha={args.alpha}.json",
        )

    save_name = (
        f"baseline_{args.algorithm}_{args.dataset}_alpha{args.alpha}_{args.client_num}clients/"
        f"{args.model}_lr{args.lr}_E{args.local_epochs}_mu{mu}_T{args.communication_rounds}_{args.tag}"
    )
    save_root = os.path.join(ROOT, "results", save_name)
    os.makedirs(save_root, exist_ok=True)

    log_path = os.path.join(save_root, "log.txt")
    if os.path.exists(log_path):
        raise SystemExit(f"日志已存在，避免覆盖: {log_path}")

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(fh)

    setup_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    logging.info("split_file: %s", split_path)
    logging.info("save_root: %s", save_root)

    dataset_info, train_set, _test_set, test_loader = get_dataset(
        args.dataset, args.dataset_root, batch_size=256
    )
    with open(split_path, "r") as f:
        split = json.load(f)
    client_indices: List[List[int]] = split["client_idx"]

    if len(client_indices) != args.client_num:
        logging.warning(
            "JSON 中 client 数 %s 与 --client_num %s 不一致，以 JSON 为准",
            len(client_indices),
            args.client_num,
        )

    num_clients = len(client_indices)
    global_model = get_model(args.model, dataset_info).to(device)

    cfg = vars(args).copy()
    cfg["mu_effective"] = mu
    cfg["split_path"] = split_path
    logging.info(cfg)

    acc, te_loss = evaluate_model(global_model, test_loader, device)
    logging.info("init (random init) test_acc=%.4f test_loss=%.6f", acc, te_loss)

    acc_list = []
    for rnd in range(args.communication_rounds):
        selected = select_client_indices(num_clients, args.join_ratio)
        global_state = copy.deepcopy(global_model.state_dict())

        local_states = []
        local_weights = []
        for k in selected:
            idx = client_indices[k]
            n_k = len(idx)
            if n_k == 0:
                continue
            subset = Subset(train_set, idx)
            loader = DataLoader(
                subset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )
            local_net = get_model(args.model, dataset_info).to(device)
            sd = local_train_one_client(
                local_net,
                global_state,
                loader,
                device,
                args.local_epochs,
                args.lr,
                args.momentum,
                args.weight_decay,
                mu,
            )
            local_states.append(sd)
            local_weights.append(n_k)

        if not local_states:
            raise RuntimeError("本轮无有效客户端更新")

        new_state = fedavg_aggregate(local_states, local_weights)
        global_model.load_state_dict(new_state)

        if rnd % args.eval_gap == 0 or rnd == args.communication_rounds - 1:
            acc, te_loss = evaluate_model(global_model, test_loader, device)
            logging.info("round %s test_acc=%.4f test_loss=%.6f clients=%s", rnd, acc, te_loss, selected)
            acc_list.append((rnd, acc, te_loss))

    logging.info("final_acc_curve: %s", acc_list)


if __name__ == "__main__":
    main()
