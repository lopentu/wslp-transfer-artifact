#!/usr/bin/env bash
# Restart wrapper for camera-ready round 1, 2026-09-13 ~17:00.
#
# Why this exists: the shared-GPU booking was released, so the queue no longer
# needs the 09:00-20:00 window. Those bounds are runtime variables, not literals,
# but the queue had already read them -- so releasing them means relaunching.
# The old launcher was killed mid-cell and its trainer deliberately left alive,
# because throwing away 8 minutes of a 22-minute cell to change a deadline is a
# bad trade. Nothing will run that cell's five evals now that its launcher is
# gone; the restarted queue picks it up as "already trained" (cell_done reads
# log.json's epochs_done) and runs them.
#
# Gate on the captured PID, never on the script name: a launcher shell carries
# the name it launched on its own command line, so pgrep matches the wrong
# process. run_cr1.sh's own guard learned this the hard way.
#
#   WAIT_PID=999099 DAY_START=00:00 PLAN_TIME=23:30 STOP_TIME=23:59 \
#     setsid nohup ./run_cr1b_launch.sh >> runs/logs/cr1.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"

WAIT_PID=${WAIT_PID:-}
if [ -n "$WAIT_PID" ] && kill -0 "$WAIT_PID" 2>/dev/null; then
  echo "### cr1 restarted without the 20:00 window; waiting for the in-flight"
  echo "### trainer (pid $WAIT_PID) so its cell is not wasted"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
  echo "### in-flight trainer finished $(date '+%F %H:%M'); its evals are the"
  echo "### restarted queue's first job"
  sleep 5
fi
exec ./.run_cr1b.running.sh
