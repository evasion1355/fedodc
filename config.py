# -*- coding: utf-8 -*-
"""FedOdc 主实验（main.py）命令行参数定义。"""
import argparse

parser = argparse.ArgumentParser(description="FedOdc 主实验")

# ---------- 随机种子与计算设备 ----------
parser.add_argument("--seed", type=int, default=19260817, help="随机种子，保证可复现")
parser.add_argument("--device", type=str, default="cuda:0", help="训练设备，如 cuda:0 或 cpu")

# ---------- 数据与划分（划分 JSON 路径由 main.py 按 dataset/client_num/alpha 自动拼接）----------
parser.add_argument(
    "--dataset_root",
    type=str,
    default="./dataset/torchvision",
    help="数据集根目录（原始图像/NPZ 等）",
)
parser.add_argument("--dataset", type=str, default="CIFAR10", help="数据集名称，如 CIFAR10、PathMNIST、OrganSMNIST")
parser.add_argument("--client_num", type=int, default=10, help="客户端数量 K，须与划分 JSON 一致")
parser.add_argument(
    "--partition",
    type=str,
    default="dirichlet",
    help="划分方式；当前 main.py 仅允许 dirichlet",
)
parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet Non-IID 参数 α，用于匹配划分文件名")

# ---------- 全局模型与联邦轮次 ----------
parser.add_argument("--model", type=str, default="ConvNet", help="全局分类网络名称，如 ConvNetBN")
parser.add_argument("--communication_rounds", type=int, default=10, help="联邦通信轮数 T")
parser.add_argument("--join_ratio", type=float, default=1.0, help="每轮参与训练的客户端比例（1.0 为全参与）")
parser.add_argument("--lr_server", type=float, default=0.01, help="服务端优化全局模型骨干的学习率")
parser.add_argument("--momentum_server", type=float, default=0.9, help="服务端 SGD 动量")
parser.add_argument("--weight_decay_server", type=float, default=0, help="服务端权重衰减（L2）")

# ---------- 服务端训练与批大小 ----------
parser.add_argument("--batch_size", type=int, default=256, help="服务端在合成集上训练的批大小")
parser.add_argument(
    "--model_epochs",
    type=int,
    default=1000,
    help="每通信轮服务端外层训练轮数（与源码 range(model_epochs+1) 对应，默认共 1001 步）",
)

# ---------- 客户端可学习合成图像（浓缩）----------
parser.add_argument("--ipc", type=int, default=10, help="每类合成图像数量（compression_ratio=0 时使用）")
parser.add_argument(
    "--compression_ratio",
    type=float,
    default=0.0,
    help="按真实样本比例进行类别内抽样时的压缩率；>0 时按该比例而非 ipc 控制每类条数",
)
parser.add_argument("--dc_iterations", type=int, default=1000, help="客户端单次浓缩迭代步数")
parser.add_argument("--dc_batch_size", type=int, default=256, help="客户端浓缩阶段对齐真实/合成嵌入时的批大小")
parser.add_argument("--image_lr", type=float, default=1.0, help="合成图像像素（及可学习参数）的学习率")
parser.add_argument("--image_momentum", type=float, default=0.5, help="合成图像优化动量")
parser.add_argument("--image_weight_decay", type=float, default=0, help="合成图像侧权重衰减")
parser.add_argument(
    "--init",
    type=str,
    default="real",
    help="合成图像初始化方式，如 real（自真实样本）",
)
parser.add_argument("--clip_norm", type=float, default=30, help="浓缩阶段梯度裁剪范数上界")
# 兼容旧命令行（无效果）；主流程固定为客户端 SWD 嵌入空间浓缩，不再做服务端门禁校验
parser.add_argument("--weighted_sample", action="store_true", help=argparse.SUPPRESS)
parser.add_argument(
    "--weighted_swd",
    "--weighted_mmd",
    action="store_true",
    dest="weighted_swd",
    help=argparse.SUPPRESS,
)
parser.add_argument("--con_beta", type=float, default=0.0, help="服务端非对称监督对比项权重；0 表示关闭")
parser.add_argument("--con_temp", type=float, default=1.0, help="对比学习温度系数")
parser.add_argument("--topk", type=int, default=3, help="双教师 logits 融合时的 Top-K 类数")
parser.add_argument("--lr_head", type=float, default=0.01, help="对比支路 Projector（头）的 Adam 学习率")
parser.add_argument("--weight_decay_head", type=float, default=0, help="对比支路 Adam 的权重衰减")
parser.add_argument("--gamma", type=float, default=1.0, help="客户端双模型 logits 融合中的缩放系数 γ")
parser.add_argument(
    "--logit_lambda",
    "--lamda",
    type=float,
    default=0.5,
    dest="logit_lambda",
    help="双模型 logits 线性融合系数 λ（--lamda 为拼写兼容别名）",
)
parser.add_argument("--b", type=float, default=0.7, help="融合后 logits 的偏置项系数 b（与 γ、λ 共同参与融合）")

# ---------- 可微增广（DSA）----------
parser.add_argument(
    "--dsa_strategy",
    type=str,
    default="color_crop_cutout_flip_scale_rotate",
    help="可微 Siamese 增广策略名；设为 None 时关闭 DSA",
)
parser.add_argument(
    "--preserve_all",
    action="store_true",
    default=False,
    help="为 True 时不按阈值丢弃历史合成缓存（默认会截断以控内存）",
)

# ---------- 日志与结果目录 ----------
parser.add_argument("--eval_gap", type=int, default=1, help="每隔多少通信轮做一次测试集评估（1 表示每轮）")
parser.add_argument(
    "--tag",
    type=str,
    default="0",
    help="结果子目录标签，用于区分多次实验（写入 results/ 下路径名）",
)
