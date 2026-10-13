#!/usr/bin/env bash
# 在本机创建项目专用 Python 环境 .venv，并安装 numpy + torch（CPU 版，用于生成参考结果、对拍）。
# 用法: scripts/setup_env.sh        之后:  source .venv/bin/activate
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV_DIR:-$ROOT/.venv}"
PKGS="${PKGS:-numpy torch}"
PYV="${PYTHON_VERSION:-3.11}"

if [ -x "$VENV/bin/python" ]; then
  echo "==> 复用已有环境: $VENV"
elif command -v uv >/dev/null 2>&1; then
  echo "==> 用 uv 创建环境 (python ${PYV}): $VENV"
  uv venv --python "$PYV" "$VENV"
else
  PY=""
  for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
      PY="$c"; break
    fi
  done
  if [ -z "$PY" ]; then
    echo "没有找到 3.10 及以上的 python3（macOS 自带的通常是 3.9，太旧）。"
    echo "任选一个安装后重跑本脚本："
    echo "  brew install uv            # 推荐，会自动下载合适的 Python"
    echo "  brew install python@3.12"
    exit 1
  fi
  echo "==> 用 $PY ($("$PY" -V)) 创建环境: $VENV"
  "$PY" -m venv "$VENV"
fi

echo "==> 安装: ${PKGS}（torch 比较大，需要几分钟）"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$VENV/bin/python" $PKGS
else
  "$VENV/bin/python" -m pip install -U pip
  "$VENV/bin/python" -m pip install $PKGS
fi

echo "==> 验证"
for p in $PKGS; do
  "$VENV/bin/python" -c "import $p; print('$p', getattr($p, '__version__', '?'))"
done
echo
echo "完成。使用方式："
echo "  source .venv/bin/activate && python ref/layernorm_ref.py"
echo "  或不激活直接运行:  .venv/bin/python ref/layernorm_ref.py"
