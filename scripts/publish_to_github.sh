#!/usr/bin/env bash
# 在本机「已登录 GitHub CLI」的前提下，一键创建公开仓库 fedodc 并推送 main。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GH="$(command -v gh || command -v /opt/homebrew/bin/gh || true)"
if [[ -z "$GH" ]]; then
  echo "未找到 gh。请先执行: brew install gh"
  exit 1
fi

if ! "$GH" auth status >/dev/null 2>&1; then
  echo "尚未登录 GitHub。请在终端执行（按提示浏览器授权）："
  echo "  $GH auth login"
  exit 1
fi

if git remote get-url origin >/dev/null 2>&1; then
  echo "远程 origin 已存在，改为直接推送: git push -u origin main"
  git push -u origin main
else
  echo "将在当前登录账号下创建公开仓库 fedodc 并推送..."
  "$GH" repo create fedodc --public --source=. --remote=origin --push
fi

echo "完成。请到 GitHub 上打开你的 fedodc 仓库复制 HTTPS 链接。"
