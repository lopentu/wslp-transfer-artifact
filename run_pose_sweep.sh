#!/usr/bin/env bash
# The encoder-only transfer axis: what transfers when the written output
# language is held constant.
#
# Every row of the main table moves two things at once. `csl_daily` and
# `how2sign` differ in sign language (CSL/ASL) *and* in the written language
# their decoder was fine-tuned to emit (Chinese/English), because that is how
# the checkpoints were released. So the 77.6 vs 45.7 gap cannot be attributed
# to sign language at all — the honest reading is that it is mostly about
# whether the decoder can still reach Chinese.
#
# `--init-parts pose` takes only the GCN pose encoder and leaves mT5 at its
# published initialisation. Run it for every source language and the decoder is
# then byte-identical across rows: vanilla mT5-base, no written language
# preferred. The only surviving difference is which sign language the encoder
# was pretrained on, which is the comparison the paper claims to be making.
#
# `--init random` is already the floor for this axis (random encoder + the same
# vanilla mT5), and csl_daily-pose is covered by `run_all.sh split`, so only the
# three ASL sources are missing.
#
# Both outcomes are publishable, which is why this is worth the card:
#   all rows == random  -> encoder-level cross-sign-language transfer is not
#                          measurable here, and the headline gain is not
#                          sign-language transfer.
#   csl_daily > ASL     -> a genuine sign-language effect, with written
#                          language controlled for the first time.
#
# Fold-major: a sweep cut off half way leaves one complete row of the axis
# behind rather than one complete fold of an incomplete axis.
#
# Fold this into run_all.sh as a `poseaxis` stage once the card is idle; it is
# separate only because run_all.sh was mid-execution when this was queued.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

INITS=${INITS:-how2sign openasl wlasl}
FOLDS=${FOLDS:-3}
PARTS=${PARTS:-pose}
TUNE=${TUNE:-full}
CTXK=${CTXK:-4}
EPOCHS=${EPOCHS:-12}
SEED=${SEED:-0}
EVAL_CONDS=${EVAL_CONDS:-local,text_only}

GPU_NEED_MIB=${GPU_NEED_MIB:-16000}
GPU_WAIT_MAX=${GPU_WAIT_MAX:-43200}

# Two different deadlines, because they do different jobs. By agreement
# (2026-08-12) a colleague has the card from 21:00 to 09:00.
#
#   PLAN_BY (20:00)  the *scheduling* limit: we never start a cell we do not
#                    expect to finish by then. This is the one that normally
#                    binds, and it stops the sweep cleanly at a cell boundary.
#   HARD_STOP (20:45) the *safety net*: a cell already running is killed only if
#                    it overruns this. Killing at PLAN_BY instead would throw
#                    away a nearly-finished cell every time an estimate slipped
#                    by a minute, which is why the two are not the same clock.
#
# Measured rate is 20-23 min/cell (the 18-cell sweep ran 04:00-10:55), so a cell
# started with 30 min left should land well before PLAN_BY, and HARD_STOP is
# only reached by a genuinely stuck run. Empty PLAN_BY disables both.
PLAN_BY=${PLAN_BY:-20:00}
HARD_STOP=${HARD_STOP:-20:45}
CELL_BUDGET_MIN=${CELL_BUDGET_MIN:-30}

PLAN_TS=""; HARD_TS=""
if [ -n "$PLAN_BY" ]; then
  # Deliberately "today": if it is already past PLAN_BY, the deadline is in the
  # past and every cell is refused. Rolling over to tomorrow would silently do
  # the opposite of what the agreement asks for.
  PLAN_TS=$(date -d "today $PLAN_BY" +%s 2>/dev/null) || {
    echo "### PLAN_BY='$PLAN_BY' is not a time I can parse — refusing to run without a deadline"; exit 1; }
  HARD_TS=$(date -d "today $HARD_STOP" +%s 2>/dev/null) || {
    echo "### HARD_STOP='$HARD_STOP' is not a time I can parse"; exit 1; }
  echo "### plan-by $PLAN_BY (no cell starts with under ${CELL_BUDGET_MIN} min left); hard stop $HARD_STOP"
fi

secs_to_plan() { [ -n "$PLAN_TS" ] || { echo 999999; return; }; echo $(( PLAN_TS - $(date +%s) )); }
secs_to_hard() { [ -n "$HARD_TS" ] || { echo 999999; return; }; echo $(( HARD_TS - $(date +%s) )); }
have_time_for_a_cell() {
  [ -n "$PLAN_TS" ] || return 0
  [ "$(secs_to_plan)" -ge $(( CELL_BUDGET_MIN * 60 )) ]
}

# Do not race our own queue for the card. The split rows were started first and
# a second job of ours arriving mid-cell would OOM one of the two, not overlap
# with it.
wait_for_our_own_queue() {
  local waited=0
  while pgrep -f 'run_all\.sh split' >/dev/null 2>&1; do
    have_time_for_a_cell || { echo "### plan-by reached while waiting for the split queue"; return 1; }
    if [ $((waited % 900)) -eq 0 ]; then
      echo "### waiting for the split queue to finish  ($(date +%H:%M))"
    fi
    sleep 60
    waited=$((waited + 60))
  done
  [ "$waited" -gt 0 ] && echo "### split queue done after $((waited / 60)) min"
  return 0
}

wait_for_gpu() {
  local waited=0 free
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge "$GPU_NEED_MIB" ]; then
      [ "$waited" -gt 0 ] && echo "### GPU free (${free} MiB) after $((waited / 60)) min"
      return 0
    fi
    # Waiting past the point where a cell could still finish by PLAN_BY just
    # holds the queue open for nothing.
    have_time_for_a_cell || { echo "### plan-by reached while waiting for a GPU window"; return 1; }
    if [ "$waited" -ge "$GPU_WAIT_MAX" ]; then
      echo "### gave up: ${free} MiB free, need ${GPU_NEED_MIB}, waited $((waited / 60)) min"
      return 1
    fi
    if [ $((waited % 900)) -eq 0 ]; then
      echo "### waiting for the GPU: ${free} MiB free, need ${GPU_NEED_MIB} MiB  ($(date +%H:%M))"
    fi
    sleep 60
    waited=$((waited + 60))
  done
}

FAILED=(); DEFERRED=()
cell() {
  local f=$1 init=$2
  local name="f${f}_local_${init}-${PARTS}_${TUNE}_k${CTXK}_s${SEED}"
  if [ -f "runs/$name/best.pt" ]; then
    echo "--- $name already trained"
  else
    if ! have_time_for_a_cell; then
      DEFERRED+=("$name"); return
    fi
    wait_for_gpu || { DEFERRED+=("$name"); return; }
    echo "--- $name  ($(date +%H:%M), $(( $(secs_to_plan) / 60 )) min to plan-by)"
    # `timeout` is the safety net, not the normal path: bounded by HARD_STOP so
    # a stuck cell cannot run into the colleague's window. -k sends KILL 60 s
    # after TERM in case the process ignores TERM.
    timeout -k 60 "$(secs_to_hard)" \
      python3 scripts/04_train.py --fold "$f" --system local --init "$init" \
      --init-parts "$PARTS" --tune "$TUNE" --ctx-k "$CTXK" --epochs "$EPOCHS" \
      --seed "$SEED"
    local rc=$?
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
      echo "### $name hit the $HARD_STOP hard stop and was killed"
      FAILED+=("hard-stop $name"); return
    elif [ "$rc" -ne 0 ]; then
      FAILED+=("train $name"); return
    fi
  fi
  [ -f "runs/$name/eval.json" ] && return 0
  timeout -k 60 "$(secs_to_hard)" python3 scripts/05_eval.py --ckpt "runs/$name/best.pt" \
    --conditions "$EVAL_CONDS" || FAILED+=("eval $name")
}

wait_for_our_own_queue || true
echo "### encoder-only axis: $INITS x $FOLDS folds, parts=$PARTS  ($(date +%H:%M))"
for f in $(seq 0 $((FOLDS - 1))); do
  for init in $INITS; do cell "$f" "$init"; done
done

if [ ${#DEFERRED[@]} -ne 0 ]; then
  echo "### ${#DEFERRED[@]} cell(s) not started — out of time before $PLAN_BY, not failures."
  echo "### Re-run this script tomorrow after 09:00; finished cells are skipped."
  printf '  %s\n' "${DEFERRED[@]}"
fi

if [ ${#FAILED[@]} -ne 0 ]; then
  echo "### ${#FAILED[@]} cell(s) failed — rerun this script to retry just these:"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
# Only say "complete" when it is. A run that stopped at the deadline with cells
# outstanding is a partial run, and reporting it as complete is how an unfinished
# axis quietly becomes a finished-looking table.
if [ ${#DEFERRED[@]} -ne 0 ]; then
  echo "### encoder-only axis PARTIAL: ${#DEFERRED[@]} cell(s) still to run  ($(date +%H:%M))"
  exit 2
fi
echo "### encoder-only axis complete  ($(date +%H:%M))"
