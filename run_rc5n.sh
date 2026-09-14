#!/usr/bin/env bash
# The referential (pointing) wrong-clip readout for the four learning-rate cells.
#
# rc5m landed `eval_clip.json` for the two head x LR cells at 06:16-06:23, so the
# head x LR interaction can now be stated in clip dependence on the CONTENT-WORD
# set. The referential set is the other instrument, and the two price video
# differently by design -- that difference is the paper's own result -- so an
# interaction reported on one instrument and silently absent on the other is a
# gap a reviewer reads as a choice. Four cells: the two intact-LR baselines and
# the two head-swap cells, so both the LR effect and the head effect at each LR
# come out on both instruments.
#
# Runs beside rc5i's text-LM ladder. That co-residency is not a guess: rc5m ran
# two of these same evals at 06:16-06:23 while the lr=3e-4 rung was training and
# both finished. The gate is still 7500 MiB, and rc5i's next rung gates on 11000,
# so the worst case is that the ladder waits out one four-minute eval.
#
#   cp run_rc5n.sh .run_rc5n.running.sh && chmod +x .run_rc5n.running.sh \
#     && nohup ./.run_rc5n.running.sh >> runs/logs/rc5_n.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

DATA=data
OUT_NAME=eval_clip_pron.json
CONDITIONS=local,blank_plain,swap_plain@0,swap_plain@1,swap_plain@2,shuffle_frames
BATCH=4
WORKERS=2
GPU_NEED_MIB=${GPU_NEED_MIB:-7500}
HOLD_FILE=${HOLD_FILE:-.gpu-hold}
# No cut-off before the submission deadline: the card is ours by agreement for
# the rest of the run. HARD_STOP is still enforced by `timeout` rather than by an
# estimate, because a wedged eval must not hold the queue open indefinitely.
PLAN_BY=${PLAN_BY:-2026-08-24 06:00}
HARD_STOP=${HARD_STOP:-2026-08-24 08:00}

PLAN_TS=$(date -d "$PLAN_BY" +%s) || exit 1
HARD_TS=$(date -d "$HARD_STOP" +%s) || exit 1
secs_to_hard() { echo $(( HARD_TS - $(date +%s) )); }
have_time() { [ "$(( PLAN_TS - $(date +%s) ))" -ge 600 ]; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

wait_for_gpu() {
  local waited=0
  while :; do
    [ -f "$HOLD_FILE" ] || { [ "$(gpu_free)" -ge "$GPU_NEED_MIB" ] && return 0; }
    have_time || return 1
    [ $(( waited % 900 )) -eq 0 ] && echo "### waiting for ${GPU_NEED_MIB} MiB ($(date '+%m-%d %H:%M'))"
    sleep 60; waited=$(( waited + 60 ))
  done
}

CELLS=(
  f0_local_how2sign_lr1e4_full_k4_s0
  f0_local_how2sign_lr3e4_full_k4_s0
  f0_local_how2sign+mt5_base-lm_head_lr1e4_full_k4_s0
  f0_local_how2sign+mt5_base-lm_head_lr3e4_full_k4_s0
)

DONE=(); FAILED=(); SKIPPED=()
echo "### rc5n starts $(date '+%F %H:%M'), ${#CELLS[@]} cells -> $OUT_NAME"
for n in "${CELLS[@]}"; do
  [ -f "runs/$n/best.pt" ] || { echo "--- $n: no best.pt"; SKIPPED+=("$n"); continue; }
  [ -f "runs/$n/$OUT_NAME" ] && { echo "--- $n already scored"; continue; }
  have_time    || { SKIPPED+=("$n"); continue; }
  wait_for_gpu || { SKIPPED+=("$n"); continue; }
  echo "--- $n ($(date '+%m-%d %H:%M'))"
  # PIPESTATUS, not the pipeline's status: the pipeline's is grep's, so a python
  # that died would look like a success whenever grep still had a line to print.
  timeout -k 60 "$(secs_to_hard)" "$PY" scripts/05_eval.py \
      --ckpt "runs/$n/best.pt" --data "$DATA" --batch "$BATCH" \
      --workers "$WORKERS" --conditions "$CONDITIONS" \
      --out "runs/$n/$OUT_NAME" 2>&1 | grep -v "^Loading weights"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ] && [ -f "runs/$n/$OUT_NAME" ]; then DONE+=("$n")
  else echo "### FAILED $n (exit $rc)"; rm -f "runs/$n/$OUT_NAME"; FAILED+=("$n"); fi
done

echo
echo "### rc5n complete $(date '+%F %H:%M'): ${#DONE[@]} scored, ${#FAILED[@]} failed, ${#SKIPPED[@]} skipped"
[ ${#FAILED[@]}  -ne 0 ] && printf 'failed:  %s\n'  "${FAILED[@]}"
[ ${#SKIPPED[@]} -ne 0 ] && printf 'skipped: %s\n' "${SKIPPED[@]}"
exit 0
