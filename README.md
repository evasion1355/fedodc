# 非独立同分布场景下基于数据浓缩的联邦图像分类

项目名称：非独立同分布场景下基于数据浓缩的联邦图像分类

英文名称：Federated Image Classification Based on Data Condensation in Non-IID Scenarios

开源地址：https://github.com/evasion1355/fedodc

发布时间：2026年5月

作者信息：
负责人：李星源，学号：2022104711，网络空间安全专业，本科

《非独立同分布场景下基于数据浓缩的联邦图像分类》实验代码，实现了面向 Non-IID 数据划分的联邦学习框架。客户端在本地使用数据集浓缩（Data Condensation）技术——通过可学习合成图像在冻结骨干的嵌入空间中利用切片 Wasserstein 距离（SWD）近似真实数据分布，服务端聚合各客户端合成图像后在历史缓存上训练全局分类模型，并支持非对称监督对比学习增强类间区分度。可供同行研究人员快速复现论文实验结果，并开展联邦学习、数据浓缩、Non-IID 场景下的二次开发。

开发语言：Python

代码规模：约 3000 行

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
    ├── generate_dirichlet_split.py
    ├── plot_thesis_partition_figures.py
    ├── run_ablation.sh
    └── compare_baselines.sh
```

## 环境配置

```bash
conda env create -f requirements.yaml
conda activate fed_dc
```

核心依赖：PyTorch 2.1 + CUDA 11.8、TorchVision、MedMNIST。

## 数据准备与 Non-IID 划分

数据集自动下载至 `./dataset/torchvision`。划分文件预置于 `dataset/split_file/`，如需重新生成：

```bash
python dataset/data/dataset_partition.py --dataset CIFAR10 --method dirichlet --client_num 10 --alpha 0.05
```

## 运行主实验

```bash
# CIFAR-10
python main.py --dataset CIFAR10 --model ConvNetBN --alpha 0.05 --client_num 10 \
  --device cuda:0 --communication_rounds 10 --compression_ratio 0.02 \
  --dc_iterations 10000 --lr_server 0.005 --tag my-run

# PathMNIST
python main.py --dataset PathMNIST --model ConvNetBN --lr_server 0.001 \
  --weight_decay_server 1e-6 --compression_ratio 0.01 --dc_iterations 5000 \
  --image_lr 0.2 --init real --clip_norm 10 --b 0 --con_beta 0.05 \
  --device cuda:0 --topk 5 --alpha 0.05 --tag 2-2-1

# OrganSMNIST
python main.py --dataset OrganSMNIST --model ConvNetBN --compression_ratio 0.05 \
  --lr_server 0.001 --dc_iterations 10000 --image_lr 0.1 --init real \
  --con_beta 0.1 --topk 5 --con_temp 0.1 --device cuda:0 --alpha 0.05 --tag 6-2-1
```

更多命令参见 `run.sh`，日志与结果保存在 `results/` 下。

## 基线方法

```bash
bash baselines/run_baselines.sh
```

提供 FedAvg 与 FedProx 基线，与主实验共享数据划分与模型配置。

## 项目声明

- 项目名称：非独立同分布场景下基于数据浓缩的联邦图像分类（FedODC）
- 项目作者：李星源
- 作者单位：天津大学网络空间安全学院
- 开发语言：Python
- 框架：PyTorch
- 核心技术：联邦学习、数据浓缩、SWD 切片 Wasserstein 距离、Non-IID 数据划分、DiffAugment 可微增广、非对称监督对比学习

## 许可证

本项目采用 MIT License。
