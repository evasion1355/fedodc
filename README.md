# FedODC · 非 IID 场景下的联邦图像分类（数据浓缩）

面向 **Non-IID（非独立同分布）** 数据划分的联邦学习实验代码：客户端在本地用 **数据集浓缩 / Data Condensation（可学习合成图像）** 近似真实分布的嵌入，服务端聚合后在合成集上更新全局图像分类模型。适用于 CIFAR-10、MedMNIST（PathMNIST / OrganSMNIST 等）等数据集。

## 开源与引用

本项目用于课程作业 / 毕业论文与同行复现实验。**若你已 fork 或引用本仓库，建议在论文或报告中注明仓库链接与作者信息。**

## 环境

推荐使用 Conda，依赖见根目录 [`requirements.yaml`](requirements.yaml)。

```bash
conda env create -f requirements.yaml
conda activate fed_dc
```

## 数据与 Non-IID 划分

1. 将数据集下载到 `./dataset/torchvision`（或 `--dataset_root` 指定路径）。
2. 使用预先生成的 `dataset/split_file/<数据集>_client_num=<K>_alpha=<α>.json`，或按需运行脚本生成 Dirichlet 划分。
3. 主程序当前仅支持 **`--partition dirichlet`**，与划分 JSON 中的 `alpha`、客户端数量一致。

## 运行主实验

在 `fedodc` 目录下：

```bash
python main.py --dataset CIFAR10 --model ConvNetBN --alpha 0.05 --client_num 10 \
  --device cuda:0 --communication_rounds 10 --compression_ratio 0.02 \
  --dc_iterations 10000 --lr_server 0.005 --tag my-run
```

更多示例命令可参考 [`run.sh`](run.sh)。日志与检查点保存在 `results/` 下；若同名 `log.txt` 已存在会报错以防止覆盖。

## 基线

`baselines/` 下提供 FedAvg / FedProx 等与主实验对齐的脚本，详见 [`baselines/run_baselines.sh`](baselines/run_baselines.sh)。

## 目录结构（概要）

| 路径 | 说明 |
|------|------|
| `main.py` | 入口：加载数据划分、构造 Client/Server、启动训练 |
| `config.py` | 命令行参数 |
| `src/client.py` | 客户端浓缩（嵌入空间对齐 / SWD 等） |
| `src/server.py` | 服务端轮次编排、聚合、全局模型训练 |
| `src/models.py`、`src/utils.py` | 模型与损失、增广 |
| `dataset/data/` | 数据集与 Non-IID 装载逻辑 |
| `dataset/split_file/` | 各数据集 × 客户端数 × α 的划分 JSON |

## 许可证

默认可附加 **MIT License**（请在仓库中选择 `LICENSE` 文件或在本目录添加后与学院要求核对）。
