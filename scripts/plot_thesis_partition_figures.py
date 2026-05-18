#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从划分 JSON 绘制论文用客户端标签分布图（灰度 + 纹理，便于黑白印刷）。

输出路径与 body.tex 中 \\includegraphics 一致（同时写入同名的 .png 与矢量 .pdf）；
划分数据与 dataset/split_file 下 *_client_num=10_alpha=0.05.json 完全一致，不重新抽样 Dirichlet。
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_cifar10_labels(dataset_root: Path) -> np.ndarray:
    """直接从 CIFAR-10 Python 版训练批读取标签，与 torchvision 索引一致。"""
    base = dataset_root / "cifar-10-batches-py"
    if not base.is_dir():
        raise FileNotFoundError(f"Missing {base}")
    labels: list[int] = []
    for i in range(1, 6):
        p = base / f"data_batch_{i}"
        with open(p, "rb") as f:
            batch = pickle.load(f, encoding="latin1")
        labels.extend(batch["labels"])
    return np.array(labels, dtype=np.int64)


def _load_medmnist_npz_labels(dataset_root: Path, fname: str) -> np.ndarray:
    p = dataset_root / fname
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}")
    with np.load(p) as z:
        y = z["train_labels"]
    return np.squeeze(y).astype(np.int64)


def _labels_for_dataset(name: str, dataset_root: Path) -> tuple[np.ndarray, int]:
    if name == "CIFAR10":
        y = _load_cifar10_labels(dataset_root)
        return y, 10
    if name == "PathMNIST":
        y = _load_medmnist_npz_labels(dataset_root, "pathmnist.npz")
        return y, 9
    if name == "OrganSMNIST":
        y = _load_medmnist_npz_labels(dataset_root, "organsmnist.npz")
        return y, 11
    raise ValueError(name)


def _load_client_dict(json_path: Path) -> dict[int, list[int]]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    clients = data["client_idx"]
    return {i: list(clients[i]) for i in range(len(clients))}


def plot_stacked_distribution_print_friendly(
    num_classes: int,
    num_users: int,
    dict_users: dict[int, list[int]],
    labels: np.ndarray,
    save_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 灰度 + 不同填充纹理，黑白打印仍可区分类别
    greys = plt.cm.Greys(np.linspace(0.28, 0.88, num_classes))
    hatch_bank = ["///", "\\\\\\", "|||", "---", "+++", "xxx", "...", "OO", "**", "//", "\\\\"]
    hatches = [hatch_bank[i % len(hatch_bank)] for i in range(num_classes)]

    label_distribution: list[list[int]] = [[] for _ in range(num_classes)]
    for client_id, client_data in dict_users.items():
        for idx in client_data:
            label_distribution[labels[idx]].append(client_id)

    fig, ax = plt.subplots(figsize=(12, 7))
    bins = np.arange(-0.5, num_users + 1.5, 1)
    _, _, patch_groups = ax.hist(
        label_distribution,
        stacked=True,
        bins=bins,
        rwidth=0.62,
        edgecolor="black",
        linewidth=0.35,
        label=[str(c) for c in range(num_classes)],
    )

    for class_i, patch_list in enumerate(patch_groups):
        g = greys[class_i]
        h = hatches[class_i]
        for p in patch_list:
            p.set_facecolor(g)
            p.set_hatch(h)
            p.set_edgecolor("black")
            p.set_linewidth(0.35)

    ax.set_xticks(np.arange(num_users))
    ax.set_xticklabels([str(c) for c in range(num_users)])
    ax.set_xlabel("Client index")
    ax.set_ylabel("Number of training samples")
    handles = [patch_groups[i][0] for i in range(num_classes)]
    ax.legend(
        handles,
        [f"class {c}" for c in range(num_classes)],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=9,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    pdf_path = save_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=REPO_ROOT / "dataset" / "torchvision",
        help="与训练一致的 MedMNIST/CIFAR 根目录",
    )
    parser.add_argument(
        "--split_dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "split_file",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT
        / "_天津大学__Tianjin_University__TJU_2022本科生毕业论文"
        / "Thesis"
        / "figures",
    )
    parser.add_argument("--client_num", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    specs = [
        ("PathMNIST", "partition_pathmnist_k10_a005.png"),
        ("CIFAR10", "partition_cifar10_k10_a005.png"),
        ("OrganSMNIST", "partition_organsmnist_k10_a005.png"),
    ]

    for ds_name, fname in specs:
        json_name = f"{ds_name}_client_num={args.client_num}_alpha={args.alpha}.json"
        json_path = args.split_dir / json_name
        if not json_path.is_file():
            print(f"Missing split file: {json_path}", file=sys.stderr)
            sys.exit(1)
        labels, num_classes = _labels_for_dataset(ds_name, args.dataset_root)
        dict_users = _load_client_dict(json_path)
        if len(dict_users) != args.client_num:
            print(f"Client count mismatch for {json_path}", file=sys.stderr)
            sys.exit(1)
        out_path = args.out_dir / fname
        plot_stacked_distribution_print_friendly(
            num_classes=num_classes,
            num_users=args.client_num,
            dict_users=dict_users,
            labels=labels,
            save_path=out_path,
        )
        print(f"Wrote {out_path} and {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()
