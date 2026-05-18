# 非独立同分布场景下基于数据浓缩的联邦图像分类

**项目名称**：非独立同分布场景下基于数据浓缩的联邦图像分类

**英文名称**：Federated Image Classification Based on Data Condensation in Non-IID Scenarios

**开源地址**：https://github.com/evasion1355/fedodc

**发布时间**：2026年5月

**作者信息**：
- 负责人：李星源，学号：2022104711，网络空间安全专业，本科

**指导教师**：甘文生，网络空间安全学院教师

---

《非独立同分布场景下基于数据浓缩的联邦图像分类》实验代码（FedODC），面向 Non-IID（非独立同分布）数据划分的联邦学习框架。客户端在本地使用数据集浓缩（Data Condensation）技术——通过可学习合成图像在冻结骨干的嵌入空间中利用切片 Wasserstein 距离（SWD）近似真实数据分布，服务端聚合各客户端合成图像后在历史缓存上训练全局分类模型，并支持非对称监督对比学习增强类间区分度。适用于 CIFAR-10、MedMNIST（PathMNIST / OrganSMNIST）等图像分类数据集，可供同行研究人员快速复现论文实验结果，并开展联邦学习、数据浓缩、Non-IID 场景下的二次开发。

**开发语言**：Python

**代码规模**：约 3000 行（含实验脚本与基线）

---

## 项目目录结构

```
fedodc/
├── main.py                 # 主入口：数据加载、Client/Server 构建、训练启动
├── config.py               # 命令行参数定义
├── run.sh                  # 示例运行命令
├── requirements.yaml       # Conda 环境依赖
├── src/
│   ├── client.py           # 客户端：SWD 数据浓缩（可学习合成图像）
│   ├── server.py           # 服务端：联邦轮次编排、原型聚合、全局模型训练
│   ├── models.py           # ConvNet 骨干网络与对比学习 Projector
│   └── utils.py            # SWD 损失、DiffAugment 可微增广、对比损失
├── dataset/
│   ├── data/
│   │   ├── dataset.py      # 数据集加载与 Non-IID 装载逻辑
│   │   └── dataset_partition.py  # Dirichlet 等 Non-IID 划分生成
│   └── split_file/         # 预生成的划分 JSON 文件
├── baselines/
│   ├── core.py             # FedAvg / FedProx 聚合与本地训练
│   ├── run_baseline.py     # 基线实验入口
│   └── run_baselines.sh    # 基线运行脚本
└── scripts/
    ├── generate_dirichlet_split.py   # 生成 Dirichlet 划分
    ├── plot_thesis_partition_figures.py  # 划分可视化
    ├── run_ablation.sh              # 消融实验脚本
    └── compare_baselines.sh         # 基线对比脚本
```

## 环境配置

推荐使用 Conda 创建环境：

```bash
conda env create -f requirements.yaml
conda activate fed_dc
```

核心依赖：PyTorch 2.1 + CUDA 11.8、TorchVision、MedMNIST、FastAI（仅数据划分使用）。

## 数据准备与 Non-IID 划分

1. 数据集将自动下载至 `./dataset/torchvision`（或通过 `--dataset_root` 指定路径）
2. 划分文件位于 `dataset/split_file/<数据集>_client_num=<K>_alpha=<α>.json`，预生成文件已包含常用配置
3. 如需重新生成划分，运行：
   ```bash
   python dataset/data/dataset_partition.py --dataset CIFAR10 --method dirichlet --client_num 10 --alpha 0.05
   ```
4. 主程序当前仅支持 `--partition dirichlet`

## 运行主实验

```bash
# CIFAR-10 示例
python main.py --dataset CIFAR10 --model ConvNetBN --alpha 0.05 --client_num 10 \
  --device cuda:0 --communication_rounds 10 --compression_ratio 0.02 \
  --dc_iterations 10000 --lr_server 0.005 --tag my-run

# PathMNIST 示例
python main.py --dataset PathMNIST --model ConvNetBN --lr_server 0.001 \
  --weight_decay_server 1e-6 --compression_ratio 0.01 --dc_iterations 5000 \
  --image_lr 0.2 --init real --clip_norm 10 --b 0 --con_beta 0.05 \
  --device cuda:0 --topk 5 --alpha 0.05 --tag 2-2-1

# OrganSMNIST 示例
python main.py --dataset OrganSMNIST --model ConvNetBN --compression_ratio 0.05 \
  --lr_server 0.001 --dc_iterations 10000 --image_lr 0.1 --init real \
  --con_beta 0.1 --topk 5 --con_temp 0.1 --device cuda:0 --alpha 0.05 --tag 6-2-1
```

更多命令参见 [`run.sh`](run.sh)。日志与结果保存在 `results/` 目录下。

## 基线方法

`baselines/` 提供 FedAvg 与 FedProx 基线，与主实验共享数据划分与模型配置：

```bash
bash baselines/run_baselines.sh
```

## 方法概述

**FedODC** 的核心流程如下：

1. **客户端浓缩**：每轮通信中，各客户端接收全局模型，冻结骨干网络，在嵌入空间中以 SWD 损失优化可学习合成图像，使其嵌入分布逼近真实数据分布。支持 DiffAugment 可微增广增强合成图像多样性。
2. **原型上行**：客户端上传合成图像及各类特征/logit 原型（按样本数加权），避免直接暴露原始数据。
3. **服务端聚合与训练**：服务端聚合各类原型，在滑动窗口缓存的历史合成集上训练全局模型，可选叠加基于上一轮原型的非对称监督对比损失。
4. **双模型 logit 融合**：客户端利用当前与历史全局模型的融合 logits 计算归一化熵，指导真实样本的加权采样。

## 许可证

本项目采用 [MIT License](LICENSE)。

## 引用

若本项目对你的研究有帮助，请引用：

```
@software{li2026fedodc,
  author  = {李星源},
  title   = {FedODC: Federated Image Classification Based on Data Condensation in Non-IID Scenarios},
  year    = {2026},
  url     = {https://github.com/evasion1355/fedodc}
}
```
