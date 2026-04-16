#!/bin/bash
# RedCrowWatch — start script
#
# Launched by cron (via install_cron.sh) or manually:
#   ./scripts/start_redcrowwatch.sh            # stream to YouTube
#   ./scripts/start_redcrowwatch.sh --preview  # local preview only
#
# If already running, this script exits silently (safe to call multiple times).

set -e

# Cron runs with a stripped PATH that excludes Homebrew.
# Export the full path so Python subprocesses (ffmpeg, etc.) can be found.
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PIDFILE="$PROJECT_DIR/.redcrowwatch.pid"
LOGFILE="$PROJECT_DIR/logs/stream.log"

# ── Already running? ───────────────────────────────────────────────────────────
if [ -f "$PIDFILE" ]; then
    PID="$(cat "$PIDFILE")"
    if kill -0 "$PID" 2>/dev/null; then
        echo "$(date '+%H:%M:%S')  RedCrowWatch already running (PID $PID) — nothing to do."
        exit 0
    fi
    rm -f "$PIDFILE"  # stale PID from a previous run
fi

# ── Start ──────────────────────────────────────────────────────────────────────
mkdir -p "$(dirname "$LOGFILE")"

MODE="${1:-}"          # pass --preview for local preview; leave empty for YouTube stream
cd "$PROJECT_DIR"

nohup /usr/local/bin/python3 stream.py $MODE >> "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"

echo "$(date '+%H:%M:%S')  RedCrowWatch started  PID=$(cat "$PIDFILE")  log=$LOGFILE"
