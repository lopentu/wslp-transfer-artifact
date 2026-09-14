#!/usr/bin/env bash
# The referential (pointing) wrong-clip readout for the four donor cells that
# only ever had the content-word one.
#
# The donor-identity result -- six starting projections on one recipient -- is
# the spine of S9 and finding 0f.3, and four of its six arms (`tsl_text`,
# `tsl_text_strong`, `gloss_lm`, `rand_head_nm`) have `eval_clip.json` and no
# `eval_clip_pron.json`. The other two (`mt5_base`, `csl_daily`) have both. So the
# comparison a reader would make across the row is available on one instrument
# and silently missing on the other, and the two instruments are exactly where
# this paper's own results diverge: rc5n showed the head factor separable from LR
# on content words and substitutable on referential pointing. A donor row read on
# one instrument would not have caught that.
#
# Same recipe as rc5n, four different cells. `tsl_text_strong` is the new one --
# Reviewer G's drift-matched instrument, trained 07:01-07:18 -- and it is the arm
# whose referential accuracy already beats the real sign-language donor's, so its
# grounding is the number the write-up most needs and least has.
#
# The card is shared with an `ollama` runner at the moment (9.3 GB of 24.5 GB at
# 08:55, not a training job of ours). 7500 MiB still clears with room; the gate
# handles it either way rather than an estimate doing so.
#
#   cp run_rc5p.sh .run_rc5p.running.sh && chmod +x .run_rc5p.running.sh \
#     && nohup ./.run_rc5p.running.sh >> runs/logs/rc5_p.log 2>&1 &
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
  f0_local_how2sign+tsl_text_strong-lm_head_full_k4_s0
  f0_local_how2sign+tsl_text-lm_head_full_k4_s0
  f0_local_how2sign+gloss_lm-lm_head_full_k4_s0
  f0_local_how2sign+rand_head_nm-lm_head_full_k4_s0
)

DONE=(); FAILED=(); SKIPPED=()
echo "### rc5p starts $(date '+%F %H:%M'), ${#CELLS[@]} cells -> $OUT_NAME"
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
echo "### rc5p complete $(date '+%F %H:%M'): ${#DONE[@]} scored, ${#FAILED[@]} failed, ${#SKIPPED[@]} skipped"
[ ${#FAILED[@]}  -ne 0 ] && printf 'failed:  %s\n'  "${FAILED[@]}"
[ ${#SKIPPED[@]} -ne 0 ] && printf 'skipped: %s\n' "${SKIPPED[@]}"
exit 0
