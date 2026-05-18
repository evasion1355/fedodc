#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成答辩 PPT 用示意图与实验图（PNG，300 dpi）。

输出目录：项目根下 docs/figures/ppt_defense/

依赖：matplotlib（仓库环境一般已有）
运行：在项目根执行（建议先建 venv 并安装 matplotlib）

  python3 -m venv .venv-figures && .venv-figures/bin/pip install matplotlib
  export MPLBACKEND=Agg MPLCONFIGDIR=$(pwd)/.mplconfig && mkdir -p \"$MPLCONFIGDIR\"
  .venv-figures/bin/python scripts/figures/gen_ppt_defense_figures.py

曲线数据：从 results/ 下现有 log.txt 解析；若缺少日志则跳过对应图并打印提示。

输出文件与 PPT 占位页对应关系见脚本末尾 main() 内注释。
"""
from __future__ import annotations

import os

# 无显示器/CI 下须用 Agg，避免默认后端崩溃
os.environ.setdefault("MPLBACKEND", "Agg")

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs" / "figures" / "ppt_defense"

# English labels only; default sans-serif is sufficient
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _save(fig, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / name
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


def fig01_data_silos() -> Path:
    """第 3 页：数据孤岛示意"""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    centers = [(2, 4), (5, 4.5), (8, 3.8), (3.5, 2), (6.5, 1.8)]
    labels = ["Hospital A", "Hospital B", "Hospital C", "Center 1", "Center 2"]
    for (x, y), lb in zip(centers, labels):
        ax.add_patch(FancyBboxPatch((x - 0.55, y - 0.35), 1.1, 0.7, boxstyle="round,pad=0.05", fc="#E3F2FD", ec="#1565C0", lw=1.5))
        ax.text(x, y, lb, ha="center", va="center", fontsize=11)
    ax.text(5, 5.5, "Medical data stays at each site (hard to pool centrally)", ha="center", fontsize=13, fontweight="bold")
    ax.text(5, 0.6, "Regulation & ethics: raw images / sensitive labels are often not allowed off-site", ha="center", fontsize=10, color="#444")
    return _save(fig, "ppt_01_data_silos.png")


def fig02_fl_overview() -> Path:
    """第 4 页：联邦学习架构简图"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((4.2, 2.3), 1.6, 1.0, boxstyle="round,pad=0.06", fc="#E8EAF6", ec="#3949AB", lw=2))
    ax.text(5, 2.8, "Central server\n(aggregate / schedule)", ha="center", va="center", fontsize=10)
    clients = [(1, 3.5), (1, 1.5), (8.5, 3.5), (8.5, 1.5)]
    for i, (x, y) in enumerate(clients):
        ax.add_patch(FancyBboxPatch((x - 0.65, y - 0.45), 1.3, 0.9, boxstyle="round,pad=0.04", fc="#E8F5E9", ec="#2E7D32", lw=1.2))
        ax.text(x, y, f"Client {i+1}\ndata stays local", ha="center", va="center", fontsize=9)
        arr = FancyArrowPatch((x + 0.35 * (1 if x < 5 else -1), y), (4.2 if x < 5 else 5.8, 2.6), arrowstyle="-|>", mutation_scale=12, color="#555", lw=1)
        ax.add_patch(arr)
    ax.text(5, 4.5, "Federated learning: data stays put; exchange updates (weights / gradients / …)", ha="center", fontsize=12, fontweight="bold")
    ax.text(5, 0.35, "Note: uplink content and privacy depend on implementation and threat model", ha="center", fontsize=9, color="#666")
    return _save(fig, "ppt_02_fl_overview.png")


def fig03_uplink_compare() -> Path:
    """第 5 页：上行对象对比小表"""
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    rows = [
        ["Aspect", "FedAvg / FedProx (baseline)", "FedOdc (ours)"],
        ["Uplink / round", "Full model state_dict", "Per-class synthetic tensors + feature / logit prototypes"],
        ["Server update", "Sample-weighted average of client weights", "SGD on (multi-round) synthetic cache → global model"],
        ["Typical local objective", "Cross-entropy on real data", "SWD: align synthetic vs. real embeddings (frozen backbone)"],
    ]
    table = ax.table(cellText=rows, loc="center", cellLoc="left", colWidths=[0.22, 0.38, 0.38])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.4)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor("#333")
        if i == 0:
            cell.set_facecolor("#3949AB")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#FAFAFA" if i % 2 else "white")
    ax.set_title("Ours vs. classic FL: uplink semantics & server operator (qualitative)", fontsize=12, pad=12)
    return _save(fig, "ppt_03_uplink_compare.png")


def _box(ax, x, y, w, h, text, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, wrap=True)


def _arrow(ax, x1, y1, x2, y2, text=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#444"))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.08, text, ha="center", fontsize=7, color="#333")


def fig04_fedodc_round() -> Path:
    """第 6 页：一轮联邦总览"""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(5.5, 6.5, "FedOdc: one communication round (matches Server.fit)", ha="center", fontsize=13, fontweight="bold")
    _box(ax, 0.4, 4.5, 3.2, 1.2, "Server\nsample clients · global model\n+ multi-round synthetic cache", "#E8EAF6", "#3949AB")
    _box(ax, 0.4, 2.5, 3.2, 1.5, "Client k\nreceive global model\nSWD condense → synth + prototypes", "#E8F5E9", "#2E7D32")
    _box(ax, 4.8, 2.5, 3.0, 1.5, "Uplink\nsynthetic tensors · proto stats", "#FFF3E0", "#EF6C00")
    _box(ax, 8.2, 2.5, 2.4, 1.5, "Server\nconcat · sliding cache\nCE deep train (+ contrast)", "#E3F2FD", "#1565C0")
    _arrow(ax, 2.0, 4.5, 2.0, 4.0, "broadcast")
    _arrow(ax, 2.0, 2.5, 2.0, 2.2)
    _arrow(ax, 3.6, 3.25, 4.75, 3.25, "")
    _arrow(ax, 7.85, 3.25, 8.15, 3.25, "")
    _box(ax, 4.5, 0.5, 4.5, 1.0, "End of round: update prev_syn_proto (class-mean of past synth. embeds)\nevaluate on test set (eval_gap)", "#F3E5F5", "#6A1B9A")
    _arrow(ax, 9.4, 2.5, 6.8, 1.5, "")
    ax.text(5.5, 0.15, "See thesis Ch.3 / source server.py, client.py", ha="center", fontsize=8, color="#666")
    return _save(fig, "ppt_04_fedodc_one_round.png")


def fig05_client_condense() -> Path:
    """第 8 页：客户端浓缩流程"""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.text(5, 3.7, "Client: SWD dataset condensation (train_swd_condense)", ha="center", fontsize=12, fontweight="bold")
    xs = [0.5, 2.8, 5.1, 7.4]
    texts = [
        "cal_loss\ndual-model logits mix\n→ entropy → weights",
        "pre_sample\nin-class softmax\nmultinomial sample",
        "frozen backbone\nembed(real)·detach\nembed(synth)",
        "SWDLoss\nbackprop updates\nsynthetic_images only",
    ]
    colors = ["#E3F2FD", "#E8F5E9", "#FFF8E1", "#FCE4EC"]
    for x, t, c in zip(xs, texts, colors):
        _box(ax, x, 1.5, 1.9, 1.6, t, c, "#333")
    for i in range(3):
        _arrow(ax, xs[i] + 1.9, 2.3, xs[i + 1], 2.3)
    ax.text(5, 0.35, "Optional: DiffAugment on real & synth (same seed)", ha="center", fontsize=9, color="#555")
    return _save(fig, "ppt_05_client_condense.png")


def fig06_server_train() -> Path:
    """第 9 页：服务端训练（主损失部分）"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.text(5, 5.1, "Server (1): synthetic set + sliding cache + CE deep training", ha="center", fontsize=12, fontweight="bold")
    _box(ax, 0.5, 3.5, 2.2, 1.0, "Per-class concat\nclient synthetics", "#E3F2FD", "#1565C0")
    _box(ax, 3.2, 3.5, 2.2, 1.0, "append to\nall_synthetic_data\n(sliding window opt.)", "#E8F5E9", "#2E7D32")
    _box(ax, 5.9, 3.5, 2.4, 1.0, "stack\nTensorDataset\n+ DataLoader", "#FFF3E0", "#EF6C00")
    _box(ax, 2.5, 1.2, 5.0, 1.5, "SGD + CrossEntropy\nouter epochs = model_epochs + 1\nStepLR; optional DSA on batch", "#F3E5F5", "#6A1B9A")
    _arrow(ax, 2.7, 4.0, 3.2, 4.0)
    _arrow(ax, 5.4, 4.0, 5.9, 4.0)
    _arrow(ax, 6.8, 3.5, 5.0, 2.7)
    ax.text(5, 0.35, "Main pipeline: no FedAvg-style weight averaging on server", ha="center", fontsize=9, color="#C62828", fontweight="bold")
    return _save(fig, "ppt_06_server_train.png")


def fig07_server_contrast() -> Path:
    """第 10 页：对比分支"""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.text(5, 4.6, "Server (2): asymmetric supervised contrastive (t>0, con_beta>0, B>1)", ha="center", fontsize=12, fontweight="bold")
    _box(ax, 0.5, 2.8, 2.0, 1.1, "batch (no DSA)\nembeddings", "#E3F2FD", "#1565C0")
    _box(ax, 3.0, 2.8, 2.3, 1.1, "Projector +\nL2 normalize", "#E8F5E9", "#2E7D32")
    _box(ax, 5.8, 2.8, 2.0, 1.1, "prev_syn_proto[y]\npositive proto row", "#FFF3E0", "#EF6C00")
    _box(ax, 8.1, 2.8, 1.5, 1.1, "Supervised\nContrastiveLoss", "#FCE4EC", "#AD1457")
    for i, x in enumerate([2.5, 5.3, 7.8]):
        _arrow(ax, [0.5, 3.0, 5.8][i] + [2.0, 2.3, 2.0][i], 3.35, [3.0, 5.8, 8.1][i], 3.35)
    ax.text(5, 1.5, "Total loss = CE + con_beta * L_con; projector Adam (see thesis & server.py)", ha="center", fontsize=9)
    ax.text(5, 0.5, "relation_class: logging / sanity only, not fed to contrastive loss", ha="center", fontsize=9, color="#555")
    return _save(fig, "ppt_07_server_contrast.png")


def fig08_prev_syn_proto() -> Path:
    """第 11 页：prev_syn_proto 累积示意（与 PPT 一致）"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.text(5, 5.1, "prev_syn_proto: per-class history of synthetics → embed → class mean → row L2 norm", ha="center", fontsize=11, fontweight="bold")
    ys = [4.2, 3.2, 2.2]
    for i, y in enumerate(ys):
        ax.add_patch(FancyBboxPatch((1, y - 0.25), 2.5, 0.5, boxstyle="round,pad=0.02", fc="#BBDEFB", ec="#1976D2"))
        ax.text(2.25, y, f"Round {i}: per-class synthetics", ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(2.25, 1.5), xytext=(2.25, 1.95), arrowprops=dict(arrowstyle="-|>", lw=1.5))
    ax.text(3.8, 1.5, "torch.cat\nall_syn_imgs_c[c]", fontsize=9)
    _box(ax, 5.5, 1.0, 3.5, 1.0, "Current global model embed\n→ class mean → prev_syn_proto\nfor contrast next round", "#C8E6C9", "#2E7D32")
    _arrow(ax, 3.5, 1.45, 5.5, 1.45)
    return _save(fig, "ppt_08_prev_syn_proto.png")


def parse_main_eval(path: Path) -> list[float] | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    pat = re.compile(r"^round (\d+) evaluation: test acc is ([\d.]+)", re.MULTILINE)
    last: dict[int, float] = {}
    for m in pat.finditer(text):
        last[int(m.group(1))] = float(m.group(2))
    if not last:
        return None
    n = max(last) + 1
    return [last[i] for i in range(n)]


def parse_baseline_round(path: Path) -> list[float] | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    pat = re.compile(r"^round (\d+) test_acc=([\d.]+)", re.MULTILINE)
    d: dict[int, float] = {}
    for m in pat.finditer(text):
        d[int(m.group(1))] = float(m.group(2))
    if not d:
        return None
    n = max(d) + 1
    return [d[i] for i in range(n)]


def fig09_main_curves() -> Path | None:
    """第 15 页：本文方法三数据集逐轮"""
    specs = [
        ("CIFAR-10", REPO_ROOT / "results/CIFAR10_alpha0.05_10clients/ConvNetBN_2.0%_10000dc_1000epochs_1-2-1/log.txt"),
        ("PathMNIST", REPO_ROOT / "results/PathMNIST_alpha0.05_10clients/ConvNetBN_1.0%_5000dc_1000epochs_2-2-1/log.txt"),
        ("OrganSMNIST", REPO_ROOT / "results/OrganSMNIST_alpha0.05_10clients/ConvNetBN_5.0%_10000dc_1000epochs_6-2-1/log.txt"),
    ]
    series = []
    for name, p in specs:
        v = parse_main_eval(p)
        if v is None:
            print(f"[skip] missing main run log: {p}")
            return None
        series.append((name, v))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    for ax, (name, y) in zip(axes, series):
        x = list(range(len(y)))
        ax.plot(x, [v * 100 for v in y], "o-", color="#1565C0", lw=2, ms=5)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel(r"Communication round $t$")
        ax.set_ylabel("Test top-1 (%)")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)
    fig.suptitle("FedOdc (ours): test accuracy after each round (from log.txt)", fontsize=12, y=1.02)
    return _save(fig, "ppt_09_curves_main.png")


def fig10_fedavg_curves() -> Path | None:
    """第 16 页左：FedAvg"""
    specs = [
        ("CIFAR-10", REPO_ROOT / "results/baseline_fedavg_CIFAR10_alpha0.05_10clients/ConvNetBN_lr0.01_E5_mu0.0_T10_fedavg_CIFAR10/log.txt"),
        ("PathMNIST", REPO_ROOT / "results/baseline_fedavg_PathMNIST_alpha0.05_10clients/ConvNetBN_lr0.01_E5_mu0.0_T10_fedavg_PathMNIST/log.txt"),
        ("OrganSMNIST", REPO_ROOT / "results/baseline_fedavg_OrganSMNIST_alpha0.05_10clients/ConvNetBN_lr0.01_E5_mu0.0_T10_fedavg_OrganSMNIST/log.txt"),
    ]
    series = []
    for name, p in specs:
        v = parse_baseline_round(p)
        if v is None:
            print(f"[skip] missing FedAvg log: {p}")
            return None
        series.append((name, v))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (name, y) in zip(axes, series):
        x = list(range(len(y)))
        ax.plot(x, [v * 100 for v in y], "s-", color="#C62828", lw=1.8, ms=4)
        ax.set_title(f"FedAvg · {name}", fontsize=10)
        ax.set_xlabel("$t$")
        ax.set_ylabel("Top-1 (%)")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)
    fig.suptitle("FedAvg baseline: test accuracy per round", fontsize=12, y=1.02)
    return _save(fig, "ppt_10_curves_fedavg.png")


def fig11_fedprox_curves() -> Path | None:
    """第 16 页右：FedProx"""
    specs = [
        ("CIFAR-10", REPO_ROOT / "results/baseline_fedprox_CIFAR10_alpha0.05_10clients/ConvNetBN_lr0.01_E5_mu0.01_T10_fedprox_CIFAR10/log.txt"),
        ("PathMNIST", REPO_ROOT / "results/baseline_fedprox_PathMNIST_alpha0.05_10clients/ConvNetBN_lr0.01_E5_mu0.01_T10_fedprox_PathMNIST/log.txt"),
        ("OrganSMNIST", REPO_ROOT / "results/baseline_fedprox_OrganSMNIST_alpha0.05_10clients/ConvNetBN_lr0.01_E5_mu0.01_T10_fedprox_OrganSMNIST/log.txt"),
    ]
    series = []
    for name, p in specs:
        v = parse_baseline_round(p)
        if v is None:
            print(f"[skip] missing FedProx log: {p}")
            return None
        series.append((name, v))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (name, y) in zip(axes, series):
        x = list(range(len(y)))
        ax.plot(x, [v * 100 for v in y], "^-", color="#6A1B9A", lw=1.8, ms=4)
        ax.set_title(f"FedProx ($\\mu$=0.01) · {name}", fontsize=10)
        ax.set_xlabel("$t$")
        ax.set_ylabel("Top-1 (%)")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)
    fig.suptitle("FedProx baseline: test accuracy per round", fontsize=12, y=1.02)
    return _save(fig, "ppt_11_curves_fedprox.png")


def fig12_bar_t9() -> Path | None:
    """第 17 页：t=9 末轮对比柱状图"""
    datasets = ["PathMNIST", "OrganSMNIST", "CIFAR-10"]
    main_paths = [
        REPO_ROOT / "results/PathMNIST_alpha0.05_10clients/ConvNetBN_1.0%_5000dc_1000epochs_2-2-1/log.txt",
        REPO_ROOT / "results/OrganSMNIST_alpha0.05_10clients/ConvNetBN_5.0%_10000dc_1000epochs_6-2-1/log.txt",
        REPO_ROOT / "results/CIFAR10_alpha0.05_10clients/ConvNetBN_2.0%_10000dc_1000epochs_1-2-1/log.txt",
    ]
    fa_paths = [
        REPO_ROOT / "results/baseline_fedavg_PathMNIST_alpha0.05_10clients/ConvNetBN_lr0.01_E5_mu0.0_T10_fedavg_PathMNIST/log.txt",
        REPO_ROOT / "results/baseline_fedavg_OrganSMNIST_alpha0.05_10clients/ConvNetBN_lr0.01_E5_mu0.0_T10_fedavg_OrganSMNIST/log.txt",
        REPO_ROOT / "results/baseline_fedavg_CIFAR10_alpha0.05_10clients/ConvNetBN_lr0.01_E5_mu0.0_T10_fedavg_CIFAR10/log.txt",
    ]
    fp_paths = [
        REPO_ROOT / "results/baseline_fedprox_PathMNIST_alpha0.05_10clients/ConvNetBN_lr0.01_E5_mu0.01_T10_fedprox_PathMNIST/log.txt",
        REPO_ROOT / "results/baseline_fedprox_OrganSMNIST_alpha0.05_10clients/ConvNetBN_lr0.01_E5_mu0.01_T10_fedprox_OrganSMNIST/log.txt",
        REPO_ROOT / "results/baseline_fedprox_CIFAR10_alpha0.05_10clients/ConvNetBN_lr0.01_E5_mu0.01_T10_fedprox_CIFAR10/log.txt",
    ]
    m, fa, fp = [], [], []
    for pm, pfa, pfp in zip(main_paths, fa_paths, fp_paths):
        vm = parse_main_eval(pm)
        vfa = parse_baseline_round(pfa)
        vfp = parse_baseline_round(pfp)
        if vm is None or vfa is None or vfp is None or len(vm) < 10 or len(vfa) < 10 or len(vfp) < 10:
            print("[skip] bar chart t=9: incomplete logs")
            return None
        m.append(vm[9] * 100)
        fa.append(vfa[9] * 100)
        fp.append(vfp[9] * 100)
    x = range(len(datasets))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - w for i in x], fa, width=w, label="FedAvg", color="#C62828", alpha=0.85)
    ax.bar(x, fp, width=w, label="FedProx ($\\mu$=0.01)", color="#6A1B9A", alpha=0.85)
    ax.bar([i + w for i in x], m, width=w, label="FedOdc (ours)", color="#1565C0", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Test top-1 (%)")
    ax.set_title(r"$t=9$: test accuracy comparison (from repo logs)", fontsize=12)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, "ppt_12_bar_t9.png")


def fig13_comm_qual() -> Path:
    """第 18 页：主实验 vs 基线 通信与更新差异（定性表）"""
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.axis("off")
    rows = [
        ["Item", "FedAvg / FedProx", "FedOdc (ours)"],
        ["Uplink / round", "Full-parameter state_dict", "Synthetic images + prototype statistics"],
        ["Server main op.", "Weighted average of client weights", "SGD on synthetic cache (model_epochs+1 outer passes)"],
        ["Compute vs. rounds", "E local epochs per round", "Heavy server epochs + client condensation (not matched)"],
        ["Privacy (informal)", "Gradients / weights may still leak", "No raw pixels; synthetics still encode distribution"],
    ]
    table = ax.table(cellText=rows, loc="center", cellLoc="left", colWidths=[0.18, 0.40, 0.40])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.35)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor("#333")
        if i == 0:
            cell.set_facecolor("#37474F")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#ECEFF1" if i % 2 else "white")
    ax.set_title("Comparison scope (qualitative): no strict bit / FLOP fairness claim", fontsize=12, pad=10)
    return _save(fig, "ppt_13_comm_qual_table.png")


def fig14_gain_t9() -> Path | None:
    """第 17 页备选：相对 FedAvg / FedProx 的末轮增益（百分点）"""
    datasets = ["PathMNIST", "OrganSMNIST", "CIFAR-10"]
    main_paths = [
        REPO_ROOT / "results/PathMNIST_alpha0.05_10clients/ConvNetBN_1.0%_5000dc_1000epochs_2-2-1/log.txt",
        REPO_ROOT / "results/OrganSMNIST_alpha0.05_10clients/ConvNetBN_5.0%_10000dc_1000epochs_6-2-1/log.txt",
        REPO_ROOT / "results/CIFAR10_alpha0.05_10clients/ConvNetBN_2.0%_10000dc_1000epochs_1-2-1/log.txt",
    ]
    fa_paths = [
        REPO_ROOT / "results/baseline_fedavg_PathMNIST_alpha0.05_10clients/ConvNetBN_lr0.01_E5_mu0.0_T10_fedavg_PathMNIST/log.txt",
        REPO_ROOT / "results/baseline_fedavg_OrganSMNIST_alpha0.05_10clients/ConvNetBN_lr0.01_E5_mu0.0_T10_fedavg_OrganSMNIST/log.txt",
        REPO_ROOT / "results/baseline_fedavg_CIFAR10_alpha0.05_10clients/ConvNetBN_lr0.01_E5_mu0.0_T10_fedavg_CIFAR10/log.txt",
    ]
    fp_paths = [
        REPO_ROOT / "results/baseline_fedprox_PathMNIST_alpha0.05_10clients/ConvNetBN_lr0.01_E5_mu0.01_T10_fedprox_PathMNIST/log.txt",
        REPO_ROOT / "results/baseline_fedprox_OrganSMNIST_alpha0.05_10clients/ConvNetBN_lr0.01_E5_mu0.01_T10_fedprox_OrganSMNIST/log.txt",
        REPO_ROOT / "results/baseline_fedprox_CIFAR10_alpha0.05_10clients/ConvNetBN_lr0.01_E5_mu0.01_T10_fedprox_CIFAR10/log.txt",
    ]
    g_fa, g_fp = [], []
    for pm, pfa, pfp in zip(main_paths, fa_paths, fp_paths):
        vm = parse_main_eval(pm)
        vfa = parse_baseline_round(pfa)
        vfp = parse_baseline_round(pfp)
        if vm is None or vfa is None or vfp is None or len(vm) < 10:
            return None
        m = vm[9] * 100
        g_fa.append(m - vfa[9] * 100)
        g_fp.append(m - vfp[9] * 100)
    x = range(len(datasets))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar([i - w / 2 for i in x], g_fa, width=w, label="Ours − FedAvg", color="#1565C0", alpha=0.88)
    ax.bar([i + w / 2 for i in x], g_fp, width=w, label="Ours − FedProx", color="#00838F", alpha=0.88)
    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Top-1 gain (percentage points)")
    ax.set_title(r"$t=9$: final-round accuracy gain vs. baselines", fontsize=12)
    ax.axhline(0, color="#999", lw=0.8)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, "ppt_14_gain_t9.png")


def main() -> None:
    paths = []
    paths.append(fig01_data_silos())
    paths.append(fig02_fl_overview())
    paths.append(fig03_uplink_compare())
    paths.append(fig04_fedodc_round())
    paths.append(fig05_client_condense())
    paths.append(fig06_server_train())
    paths.append(fig07_server_contrast())
    paths.append(fig08_prev_syn_proto())
    paths.append(fig13_comm_qual())
    for p in paths:
        print("written:", p)

    opt = [
        fig09_main_curves(),
        fig10_fedavg_curves(),
        fig11_fedprox_curves(),
        fig12_bar_t9(),
        fig14_gain_t9(),
    ]
    for p in opt:
        if p:
            print("written:", p)

    print("\nDone. Output directory:", OUT_DIR)
    print(
        "\nSuggested slide mapping (filename → topic):\n"
        "  ppt_01_data_silos.png       → data silos\n"
        "  ppt_02_fl_overview.png      → FL overview\n"
        "  ppt_03_uplink_compare.png   → uplink comparison table\n"
        "  ppt_04_fedodc_one_round.png → one-round pipeline\n"
        "  ppt_05_client_condense.png  → client SWD condensation\n"
        "  ppt_06_server_train.png     → server CE training\n"
        "  ppt_07_server_contrast.png  → contrastive branch\n"
        "  ppt_08_prev_syn_proto.png   → prev_syn_proto\n"
        "  ppt_09_curves_main.png      → ours, per-round curves\n"
        "  ppt_10_curves_fedavg.png    → FedAvg curves\n"
        "  ppt_11_curves_fedprox.png   → FedProx curves\n"
        "  ppt_12_bar_t9.png           → bar chart at t=9\n"
        "  ppt_14_gain_t9.png          → gain vs. baselines (optional)\n"
        "  ppt_13_comm_qual_table.png  → comm / compute scope table\n"
    )


if __name__ == "__main__":
    main()
