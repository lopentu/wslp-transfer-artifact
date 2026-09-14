#!/usr/bin/env bash
# The internal view of collapse (scripts/20_repr_collapse.py), on the cells whose
# behaviour the paper contrasts. Lowest priority of the three GPU jobs: it adds an
# appendix, where the other two decide claims in the body, so it waits for room
# rather than competing for it.
# ---------------------------------------------------------------- GPU sharing
#
# CHANGED 8/19: the colleague has released the card through the submission, so
# there is no nightly hand-back and no 21:00 boundary until 2026-08-21 20:00.
# The deadline below is therefore an ABSOLUTE timestamp rather than "today
# 20:00", and EARLIEST is empty so the queue never parks waiting for a morning
# that has already been granted. Two consequences worth knowing:
#
#   * `date -d "$PLAN_BY"`, not `date -d "today $PLAN_BY"` -- the latter cannot
#     parse a full date, and would have silently produced an unparseable
#     deadline, which these scripts treat as a refusal to run.
#   * a queue that runs past midnight used to sleep until the next 09:00,
#     because `today 09:00` is in the future at 00:30. With EARLIEST empty the
#     hold is skipped entirely.
#
# The free-memory gates stay. They are what stops us walking into somebody
# else's job, and that risk did not go away with the schedule.
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
PLAN_BY=${PLAN_BY:-2026-08-21 20:00}
HARD_STOP=${HARD_STOP:-2026-08-21 20:45}
NEED_MIB=${NEED_MIB:-9000}
HOLD_FILE=${HOLD_FILE:-.gpu-hold}
CELLS=${CELLS:-f0_local_csl_daily_full_k4_s0,f0_local_how2sign_full_k4_s0,f0_local_csl_stage1_full_k4_s0,f0_local_random_full_k4_s0,f0_local_how2sign-pose+csl_daily-mt5_full_k4_s0,f0_local_csl_daily-pose+how2sign-mt5_full_k4_s0,f0_local_openasl_full_k4_s0,f0_local_wlasl_full_k4_s0}

PLAN_TS=$(date -d "$PLAN_BY" +%s) || exit 1
HARD_TS=$(date -d "$HARD_STOP" +%s) || exit 1

waited=0
while :; do
  [ "$(date +%s)" -ge "$PLAN_TS" ] && { echo "### plan-by $PLAN_BY reached, never got room"; exit 0; }
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ ! -f "$HOLD_FILE" ] && [ "${free:-0}" -ge "$NEED_MIB" ]; then break; fi
  [ $(( waited % 1800 )) -eq 0 ] && echo "### waiting for ${NEED_MIB} MiB (have ${free:-?})  ($(date +%H:%M))"
  sleep 120; waited=$(( waited + 120 ))
done

echo "### representation pass  ($(date +%H:%M))"
timeout -k 60 $(( HARD_TS - $(date +%s) )) "$PY" scripts/20_repr_collapse.py --cells "$CELLS"
