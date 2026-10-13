#!/usr/bin/env bash
# 建立一条 8 小时的复用 SSH 连接。这里会提示输入密码，由你本人输入。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/config.sh"

if ssh -O check "$REMOTE_HOST" 2>/dev/null; then
  echo "已经连接，无需重复登录。"; exit 0
fi
if ! ssh -G "$REMOTE_HOST" 2>/dev/null | grep -qiE '^controlmaster (auto|yes)'; then
  echo "${REMOTE_HOST} 还没有启用连接复用。请把对应的 snippet 追加到 ~/.ssh/config 末尾，例如："
  echo "  cat $ROOT/scripts/ssh_config_4p6000.snippet >> ~/.ssh/config     # p6000 目标"
  echo "  cat $ROOT/scripts/ssh_config.snippet >> ~/.ssh/config            # cse 目标"
  exit 1
fi
echo "即将登录 ${REMOTE_HOST}，请输入密码（只需这一次）..."
ssh -o ControlMaster=auto -o ControlPersist=8h "$REMOTE_HOST" true \
  && ssh -O check "$REMOTE_HOST" \
  && echo "连接已建立，接下来 8 小时内 run.sh / doctor.sh 不会再要密码。"
