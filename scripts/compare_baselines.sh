#!/usr/bin/env bash
# 仅与 FedProx、FedAvg 对比：同一 α / 压缩率 / 轮数 / local_ep 下依次跑三者，看各次 log.txt 的 test acc
# 用法: bash scripts/compare_baselines.sh path
#       bash scripts/compare_baselines.sh organ cuda 15 0.05
set -euo pipefail
DS="${1:?dataset: path|organ|cifar}"
DEV="${2:-cuda}"
ROUNDS="${3:-15}"
ALPHA="${4:-0.05}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export ROUNDS ALPHA
for M in fedprox fedavg medfl; do
  echo "========== $DS | $M | alpha=$ALPHA rounds=$ROUNDS =========="
  bash run.sh "$DS" "$M" "$DEV" "$ROUNDS" "$ALPHA"
done
