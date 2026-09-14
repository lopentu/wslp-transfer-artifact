#!/usr/bin/env bash
# Clip evals for the two head + learning-rate cells (A-B1 / A-W6).
#
# Block 5 of run_rc5.sh marks `how2sign+mt5_base-lm_head_lr{1e4,3e4}` as `lex`,
# so they get eval_lex.json and no eval_clip.json. That was the right call when
# the LR sweep was a hyperparameter check. It is the wrong call now: at 3e-4 the
# INTACT checkpoint gains +13.4 points of clip dependence with no surgery at all,
# so the question the sweep now answers is whether the learning rate and the
# neutral head produce grounding by the same route -- and that question is only
# askable on Delta local-swap, which those two cells would not have.
#
# Gated on the training queue's own pid, and nothing else. Not the launcher's:
# Claude Code's wrapper shell is the queue's parent, gets orphaned to init with
# its command already finished, and never exits -- on 8/23 that silently
# dead-locked run_rc5i.sh for five hours. Resolve the queue with
#   ps -eo pid,args | awk '$2=="bash" && $3=="./.run_rc5x.running.sh" {print $1}'
#
#   cp run_rc5m.sh .run_rc5m.running.sh && \
#     WAIT_PIDS="1227517" nohup ./.run_rc5m.running.sh \
#       > runs/logs/rc5_m.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

WAIT_PIDS=${WAIT_PIDS:-}
PLAN_BY=${PLAN_BY:-2026-08-24 06:00}
HARD_STOP=${HARD_STOP:-2026-08-24 08:00}
GPU_EVAL_MIB=${GPU_EVAL_MIB:-8000}
HOLD_FILE=${HOLD_FILE:-.gpu-hold}
CONDS=local,blank_plain,swap_plain@0,swap_plain@1,swap_plain@2,shuffle_frames

PLAN_TS=$(date -d "$PLAN_BY" +%s) || exit 1
HARD_TS=$(date -d "$HARD_STOP" +%s) || exit 1
secs_to_hard() { echo $(( HARD_TS - $(date +%s) )); }
have_time() { [ "$(( PLAN_TS - $(date +%s) ))" -ge 420 ]; }
gpu_free()  { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

wait_gpu() {
  local need=${1:-$GPU_EVAL_MIB} waited=0
  while :; do
    [ -f "$HOLD_FILE" ] || { [ "$(gpu_free)" -ge "$need" ] && return 0; }
    have_time || return 1
    [ $(( waited % 900 )) -eq 0 ] && echo "### waiting for ${need} MiB ($(date +%H:%M))"
    sleep 60; waited=$(( waited + 60 ))
  done
}

# epochs_done, not best.pt: best.pt is rewritten every epoch.
cell_done() {
  local n=$1 got
  [ -f "runs/$n/best.pt" ] && [ -f "runs/$n/log.json" ] || return 1
  got=$("$PY" -c "import json;print(json.load(open('runs/$n/log.json'))['epochs_done'])" 2>/dev/null) || return 1
  [ "${got:-0}" -ge 12 ]
}

if [ -n "$WAIT_PIDS" ]; then
  echo "### waiting on queue pid(s): $WAIT_PIDS ($(date '+%F %H:%M'))"
  while :; do
    alive=0
    for p in $WAIT_PIDS; do [ -d "/proc/$p" ] && alive=1; done
    [ "$alive" -eq 0 ] && break
    have_time || { echo "### out of time while waiting"; exit 1; }
    sleep 120
  done
  echo "### gate cleared ($(date '+%F %H:%M'))"
fi

FAILED=(); SKIPPED=()
for n in f0_local_how2sign+mt5_base-lm_head_lr1e4_full_k4_s0 \
         f0_local_how2sign+mt5_base-lm_head_lr3e4_full_k4_s0; do
  [ -f "runs/$n/eval_clip.json" ] && { echo "--- $n clip already done"; continue; }
  cell_done "$n" || { echo "--- $n not finished training, skipping"; SKIPPED+=("$n"); continue; }
  wait_gpu || { SKIPPED+=("$n"); break; }
  echo "--- clip $n ($(date +%H:%M))"
  timeout -k 60 "$(secs_to_hard)" "$PY" scripts/05_eval.py \
    --ckpt "runs/$n/best.pt" --data data/lex --batch 4 --workers 2 \
    --conditions "$CONDS" --out "runs/$n/eval_clip.json" 2>&1 | grep -v "^Loading weights"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { rm -f "runs/$n/eval_clip.json"; FAILED+=("$n"); }
done

echo
[ ${#SKIPPED[@]} -ne 0 ] && { echo "### skipped:"; printf '  %s\n' "${SKIPPED[@]}"; }
[ ${#FAILED[@]}  -ne 0 ] && { echo "### failed:";  printf '  %s\n' "${FAILED[@]}"; exit 1; }
echo "### rc5m complete ($(date '+%F %H:%M'))"
