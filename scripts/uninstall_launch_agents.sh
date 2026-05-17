#!/usr/bin/env bash
set -euo pipefail

AGENT_DIR="$HOME/Library/LaunchAgents"
USER_DOMAIN="gui/$(id -u)"

PLISTS=(
  "com.ronen.bot-trad.candles-stocks.plist"
  "com.ronen.bot-trad.tradebot-main.plist"
  "com.ronen.bot-trad.tradebot-main-7.plist"
)

LABELS=(
  "com.ronen.bot-trad.candles-stocks"
  "com.ronen.bot-trad.tradebot-main"
  "com.ronen.bot-trad.tradebot-main-7"
)

for i in "${!PLISTS[@]}"; do
  plist="${PLISTS[$i]}"
  label="${LABELS[$i]}"
  dest="$AGENT_DIR/$plist"

  launchctl bootout "$USER_DOMAIN" "$dest" >/dev/null 2>&1 || true
  launchctl bootout "$USER_DOMAIN/$label" >/dev/null 2>&1 || true
  rm -f "$dest"
  echo "Uninstalled: $label"
done
