#!/usr/bin/env bash
# 把本机的 data/ 同步到集群（数据较大，所以和代码同步分开，只在重新生成数据后运行）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/config.sh"
require_master || exit 1

[ -d "$ROOT/data" ] || { echo "没有 data/。请先运行: python ref/gen_data.py"; exit 1; }
rsh <<EOF
mkdir -p $(printf %q "$REMOTE_DIR")/data
EOF
rsync -az "$ROOT/data/" "$REMOTE_HOST:$REMOTE_DIR/data/"
echo "已同步 data/ -> ${REMOTE_HOST}:${REMOTE_DIR}/data/"
