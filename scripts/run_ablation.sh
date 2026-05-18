#!/usr/bin/env bash
# 三数据集 × 基线 / MedFL / 消融（与 run.sh 默认 α=0.05、压缩率一致）
# 用法: bash scripts/run_ablation.sh cuda 15 0.05
set -euo pipefail
DEVICE="${1:-cuda}"
ROUNDS="${2:-15}"
ALPHA="${3:-0.05}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

run_one() {
  local ds="$1"
  local mode="$2"
  shift 2
  bash run.sh "$ds" "$mode" "$DEVICE" "$ROUNDS" "$ALPHA" "$@"
}

for DS in path organ cifar; do
  echo "========== ${DS} | fedavg =========="
  run_one "$DS" fedavg || true
  echo "========== ${DS} | fedprox =========="
  run_one "$DS" fedprox || true
  echo "========== ${DS} | medfl full =========="
  run_one "$DS" medfl || true
  echo "========== ${DS} | medfl w/o entropy =========="
  run_one "$DS" medfl_ab_entropy || true
  echo "========== ${DS} | medfl w/o SWD (MMD/L2) =========="
  run_one "$DS" medfl_ab_swd || true
  echo "========== ${DS} | medfl w/o DWCL (SupCon) =========="
  run_one "$DS" medfl_ab_dwcl || true
done

echo "全部任务已尝试执行。若某次因 log 已存在而报错，请换 --tag 或删对应 results 子目录。"
