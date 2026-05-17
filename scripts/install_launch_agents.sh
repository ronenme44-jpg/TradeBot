#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHD_DIR="$PROJECT_DIR/launchd"
AGENT_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$PROJECT_DIR/logs/launchd"
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

mkdir -p "$AGENT_DIR" "$LOG_DIR"

for i in "${!PLISTS[@]}"; do
  plist="${PLISTS[$i]}"
  label="${LABELS[$i]}"
  src="$LAUNCHD_DIR/$plist"
  dest="$AGENT_DIR/$plist"

  if [[ ! -f "$src" ]]; then
    echo "Missing plist: $src" >&2
    exit 1
  fi

  if launchctl print "$USER_DOMAIN/$label" >/dev/null 2>&1; then
    launchctl bootout "$USER_DOMAIN" "$dest" >/dev/null 2>&1 || true
  fi

  sed "s#__PROJECT_DIR__#$PROJECT_DIR#g" "$src" > "$dest"
  chmod 644 "$dest"

  launchctl bootstrap "$USER_DOMAIN" "$dest"
  launchctl enable "$USER_DOMAIN/$label" >/dev/null 2>&1 || true
  echo "Installed and started: $label"
done

echo
echo "Logs:"
echo "  $LOG_DIR"
echo
echo "Check status with:"
echo "  $PROJECT_DIR/scripts/status_launch_agents.sh"
