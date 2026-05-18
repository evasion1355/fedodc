#!/usr/bin/env python3
"""
不依赖 medmnist 包：从本地 .npz（MedMNIST）或 CIFAR-10 pickle 读标签，
生成与 dataset_partition.py 相同格式的 Dirichlet 划分 JSON。
用法（在项目根目录）:
  python scripts/generate_dirichlet_split.py --dataset PathMNIST --alpha 0.4 --client_num 10 \\
    --data_root ./dataset/torchvision --seed 19260817
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

# npz 文件名与 MedMNIST 下载一致
_NPZ = {
    "PathMNIST": "pathmnist.npz",
    "OrganSMNIST": "organsmnist.npz",
}


def _load_labels(dataset: str, data_root: str) -> np.ndarray:
    if dataset in _NPZ:
        path = os.path.join(data_root, _NPZ[dataset])
        if not os.path.isfile(path):
            print(f"缺少数据文件: {path}", file=sys.stderr)
            print("请先运行训练以下载 MedMNIST，或将 .npz 放到 data_root。", file=sys.stderr)
            sys.exit(1)
        z = np.load(path)
        y = np.squeeze(z["train_labels"]).astype(np.int64)
        return y
    if dataset == "CIFAR10":
        cifar_dir = os.path.join(data_root, "cifar-10-batches-py")
        if not os.path.isdir(cifar_dir):
            print(f"缺少 CIFAR-10 目录: {cifar_dir}", file=sys.stderr)
            sys.exit(1)
        labels = []
        for i in range(1, 6):
            with open(os.path.join(cifar_dir, f"data_batch_{i}"), "rb") as f:
                d = pickle.load(f, encoding="bytes")
            labels.extend(d[b"labels"])
        return np.array(labels, dtype=np.int64)
    print(f"不支持的 dataset: {dataset}", file=sys.stderr)
    sys.exit(1)


def dirichlet_split(
    labels: np.ndarray,
    num_classes: int,
    client_num: int,
    alpha: float,
    rng: np.random.Generator,
) -> dict[int, list[int]]:
    min_size = -1
    min_require_size = 0
    K = num_classes
    N = labels.shape[0]
    dict_users: dict[int, list[int]] = {}
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(client_num)]
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            rng.shuffle(idx_k)
            proportions = rng.dirichlet(np.repeat(alpha, client_num))
            proportions = np.array(
                [p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            splits = np.split(idx_k, proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, splits)]
        min_size = min(len(idx_j) for idx_j in idx_batch)
    for j in range(client_num):
        rng.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]
    return dict_users


def client_classes_from_split(
    dict_users: dict[int, list[int]], labels: np.ndarray, min_count: int = 10
) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for j, idx in dict_users.items():
        if not idx:
            out[j] = []
            continue
        unq, cnt = np.unique(labels[np.array(idx)], return_counts=True)
        out[j] = [int(c) for c, n in zip(unq, cnt) if n >= min_count]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--client_num", type=int, default=10)
    p.add_argument("--seed", type=int, default=19260817)
    p.add_argument("--data_root", type=str, default="./dataset/torchvision")
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="默认: <repo>/dataset/split_file",
    )
    args = p.parse_args()

    num_classes = {"PathMNIST": 9, "OrganSMNIST": 11, "CIFAR10": 10}[args.dataset]
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = args.out_dir or os.path.join(repo_root, "dataset", "split_file")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    labels = _load_labels(args.dataset, os.path.expanduser(args.data_root))
    dict_users = dirichlet_split(labels, num_classes, args.client_num, args.alpha, rng)
    dict_classes = client_classes_from_split(dict_users, labels)

    name = f"{args.dataset}_client_num={args.client_num}_alpha={args.alpha}.json"
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        json.dump(
            {
                "client_idx": [dict_users[i] for i in range(args.client_num)],
                "client_classes": [dict_classes[i] for i in range(args.client_num)],
            },
            f,
            indent=4,
        )
    print(f"已写入: {path}")


if __name__ == "__main__":
    main()
