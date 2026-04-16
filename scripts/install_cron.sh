#!/bin/bash
# RedCrowWatch — install cron schedule
#
# Edit the time variables below, then run this script ONCE:
#   bash scripts/install_cron.sh
#
# To see the installed schedule:   crontab -l
# To remove the schedule:          crontab -e  (delete the redcrowwatch lines)
# To re-run immediately:           ./scripts/start_redcrowwatch.sh
#
# NOTE: cron only fires when your Mac is AWAKE AND LOGGED IN.
#       If the machine is asleep at the scheduled time, the job is simply skipped
#       until the next occurrence.  This is normal and expected.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
START_SCRIPT="$PROJECT_DIR/scripts/start_redcrowwatch.sh"
STOP_SCRIPT="$PROJECT_DIR/scripts/stop_redcrowwatch.sh"
LOGFILE="$PROJECT_DIR/logs/cron.log"

# ── Window 1: Morning (standard 5-field cron: min hour day month weekday) ─────
MORNING_START="0 7  * * 1-5"   # 7:00 AM  Mon–Fri
MORNING_STOP="0 9  * * 1-5"    # 9:00 AM  Mon–Fri

# ── Window 2: Afternoon ────────────────────────────────────────────────────────
AFTERNOON_START="30 14 * * 1-5" # 2:30 PM  Mon–Fri
AFTERNOON_STOP="30 17 * * 1-5"  # 5:30 PM  Mon–Fri
# ──────────────────────────────────────────────────────────────────────────────

chmod +x "$START_SCRIPT" "$STOP_SCRIPT"
mkdir -p "$(dirname "$LOGFILE")"

# Merge new entries with existing crontab, removing any old RedCrowWatch lines first
(
    crontab -l 2>/dev/null | grep -v "redcrowwatch"
    echo "# RedCrowWatch — auto-managed by install_cron.sh"
    echo "# Morning window"
    echo "$MORNING_START   $START_SCRIPT >> $LOGFILE 2>&1"
    echo "$MORNING_STOP    $STOP_SCRIPT  >> $LOGFILE 2>&1"
    echo "# Afternoon window"
    echo "$AFTERNOON_START $START_SCRIPT >> $LOGFILE 2>&1"
    echo "$AFTERNOON_STOP  $STOP_SCRIPT  >> $LOGFILE 2>&1"
) | crontab -

echo ""
echo "Cron schedule installed:"
echo "  Morning   start : $MORNING_START   →  $START_SCRIPT"
echo "  Morning   stop  : $MORNING_STOP    →  $STOP_SCRIPT"
echo "  Afternoon start : $AFTERNOON_START →  $START_SCRIPT"
echo "  Afternoon stop  : $AFTERNOON_STOP  →  $STOP_SCRIPT"
echo ""
echo "Current crontab:"
crontab -l
echo ""
echo "Cron log will be written to: $LOGFILE"
echo "Stream log:                  $PROJECT_DIR/logs/stream.log"
