#!/usr/bin/env bash
# The referential (pointing-sign) wrong-clip reading of the five output-projection
# cells. Findings 0d-0f are all stated on the content-word instrument, and A-Q5's
# whole point is that the two instruments disagree about seed stability -- so
# "the abolition cell's clip dependence is zero" is currently a claim about one
# item set, on cells that were scored on the other set only because they did not
# exist when the 8/19 referential sweep ran.
#
# Chained on an explicit PID rather than on a script filename: Claude Code's
# wrapper shell persists after launching a queue and its command line quotes the
# snapshot's name, so `pgrep -f .run_rc5j.running.sh` matches the launcher and a
# filename gate never clears. WAIT_PIDS is captured at launch time.
#
# And it must be the QUEUE's pid ALONE, not the queue plus its launcher. The
# received wisdom -- "gate on both when unsure, a superset only waits longer" --
# is false here: that wrapper shell (the parent, lower pid) can outlive the queue
# it started, so including it turns the gate into a deadlock rather than a longer
# wait. First launch of this script used `WAIT_PIDS="1337305 1337308"` and had to
# be killed and relaunched with the child alone. Resolve it as
#   ps -eo pid,args | awk '$2=="bash" && $3=="./.run_rc5j.running.sh" {print $1}'
# which returns the queue and never the wrapper.
#
#   cp run_rc5k.sh .run_rc5k.running.sh && \
#     WAIT_PIDS="1337308" nohup ./.run_rc5k.running.sh \
#       > runs/logs/rc5_k.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"

WAIT_PIDS=${WAIT_PIDS:-}
PLAN_BY=${PLAN_BY:-2026-08-24 06:00}
PLAN_TS=$(date -d "$PLAN_BY" +%s) || exit 1

if [ -n "$WAIT_PIDS" ]; then
  echo "### waiting on PIDs: $WAIT_PIDS ($(date '+%F %H:%M'))"
  while :; do
    alive=0
    for p in $WAIT_PIDS; do [ -d "/proc/$p" ] && alive=1; done
    [ "$alive" -eq 0 ] && break
    [ "$(( PLAN_TS - $(date +%s) ))" -ge 600 ] || { echo "### out of time while waiting"; exit 1; }
    sleep 60
  done
  echo "### gate cleared ($(date '+%F %H:%M'))"
fi

# A referential cell is ~4x a lexical one, so 5 cells is ~40 min. The 8000 MiB
# gate and the timeout live in run_wrongclip.sh; this only sets the scope.
SCOPE=heads PRON=1 PLAN_BY="$PLAN_BY" HARD_STOP="${HARD_STOP:-2026-08-24 08:00}" \
  CELL_BUDGET_MIN=${CELL_BUDGET_MIN:-10} bash run_wrongclip.sh
rc=$?
echo "### rc5k complete rc=$rc ($(date '+%F %H:%M'))"
exit $rc
