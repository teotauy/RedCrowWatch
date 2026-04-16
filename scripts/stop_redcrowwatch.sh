#!/bin/bash
# RedCrowWatch — stop script
#
# Sends SIGTERM so the stream shuts down cleanly (ffmpeg pipe is closed, etc.).
# Called by cron at the scheduled stop time, or manually.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PIDFILE="$PROJECT_DIR/.redcrowwatch.pid"

if [ ! -f "$PIDFILE" ]; then
    echo "$(date '+%H:%M:%S')  No PID file found — RedCrowWatch is not running."
    exit 0
fi

PID="$(cat "$PIDFILE")"

if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    echo "$(date '+%H:%M:%S')  RedCrowWatch stopped  (PID $PID)"
    rm -f "$PIDFILE"
else
    echo "$(date '+%H:%M:%S')  PID $PID not found — already stopped."
    rm -f "$PIDFILE"
fi
