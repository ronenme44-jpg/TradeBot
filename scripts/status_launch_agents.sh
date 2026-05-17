#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_DOMAIN="gui/$(id -u)"

LABELS=(
  "com.ronen.bot-trad.candles-stocks"
  "com.ronen.bot-trad.tradebot-main"
  "com.ronen.bot-trad.tradebot-main-7"
)

echo "LaunchAgents status:"
for label in "${LABELS[@]}"; do
  if ! output="$(launchctl print "$USER_DOMAIN/$label" 2>/dev/null)"; then
    printf '  %-42s %s\n' "$label" "not loaded"
    continue
  fi

  state="$(awk -F'= ' '/state =/ {print $2; exit}' <<<"$output" | tr -d ';')"
  pid="$(awk -F'= ' '/pid =/ {print $2; exit}' <<<"$output" | tr -d ';')"
  last_exit="$(awk -F'= ' '/last exit code =/ {print $2; exit}' <<<"$output" | tr -d ';')"

  [[ -n "$state" ]] || state="loaded"
  [[ -n "$pid" ]] || pid="-"
  [[ -n "$last_exit" ]] || last_exit="-"

  printf '  %-42s state=%-10s pid=%-8s last_exit=%s\n' "$label" "$state" "$pid" "$last_exit"
done

echo
echo "Recent log files:"
find "$PROJECT_DIR/logs/launchd" -type f -maxdepth 1 -print 2>/dev/null | sort
