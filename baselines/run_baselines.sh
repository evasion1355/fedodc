#!/usr/bin/env bash
# 与 run.sh 相同数据划分（alpha=0.05, 10 clients），FedAvg / FedProx 基线。
# 在项目根目录执行: bash baselines/run_baselines.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ROUNDS="${ROUNDS:-10}"
LOCAL_E="${LOCAL_E:-5}"
LR="${LR:-0.01}"
DEVICE="${DEVICE:-cuda:0}"

run_one() {
  local algo="$1"
  local ds="$2"
  local mu="${3:-0.01}"
  local tag="${algo}_${ds}"
  python -m baselines.run_baseline \
    --algorithm "$algo" \
    --dataset "$ds" \
    --model ConvNetBN \
    --alpha 0.05 \
    --client_num 10 \
    --communication_rounds "$ROUNDS" \
    --local_epochs "$LOCAL_E" \
    --lr "$LR" \
    --batch_size 64 \
    --device "$DEVICE" \
    --mu "$mu" \
    --tag "$tag"
}

# FedAvg
run_one fedavg PathMNIST 0
run_one fedavg OrganSMNIST 0
run_one fedavg CIFAR10 0

# FedProx（μ 可按文献网格再扫：0.001, 0.01, 0.1）
run_one fedprox PathMNIST 0.01
run_one fedprox OrganSMNIST 0.01
run_one fedprox CIFAR10 0.01
