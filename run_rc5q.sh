#!/usr/bin/env bash
# The last nine cells with no referential wrong-clip reading, which between them
# are every remaining `\HeadClip*Pron` macro in numbers.tex.
#
# rc5p closed the donor row. This closes the rest, and the reason is the one rc5n
# found the hard way: the two item sets can disagree. On the head x LR interaction
# they did -- +13.6 on content words against +5.1 (p = 0.074) on pointing -- so a
# result that exists on one instrument and not the other is a result nobody has
# actually checked. Three separate findings are in that state:
#
#   O-P3's scale control   how2sign+rand_head-lm_head (library init, not norm-matched)
#                          csl_daily+rand_head_nm-lm_head (the CSL-side twin)
#   A-B3's rescales        how2sign+how2sign_{cjk,cjkgl,gl}-lm_head
#   O-P2's replication     {openasl,wlasl}+{mt5_base,csl_daily}-lm_head
#
# Cost: nine evals. rc5p measured 2m09s-2m41s each, not the 4 min the older
# estimate assumed, so ~25 min.
#
# Same GPU arrangement as rc5p's relaunch, and for the same reason: richard's
# `ollama serve` is a persistent model server (up 1d16h at 11:00), so `.gpu-hold`
# is not going to clear on its own today. Kevin authorised bypassing it WITH
# HEADROOM. HOLD_FILE points at a path that does not exist; GPU_NEED_MIB is 13000
# against a measured 3.3 GB of actual use, re-checked before every cell, so
# ollama keeps >= 9.7 GB of room and a growing neighbour parks the remaining
# cells instead of colliding with them. `.gpu-hold` itself is left alone -- the
# override is scoped to this queue, not to the machine.
#
#   cp run_rc5q.sh .run_rc5q.running.sh && chmod +x .run_rc5q.running.sh
#   HOLD_FILE=.gpu-hold.bypassed-for-rc5q GPU_NEED_MIB=13000 \
#     nohup ./.run_rc5q.running.sh >> runs/logs/rc5_q.log 2>&1 &
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
# No cut-off before the submission deadline. HARD_STOP is still enforced by
# `timeout` rather than by an estimate, because a wedged eval must not hold the
# queue open indefinitely.
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
  f0_local_csl_daily+rand_head_nm-lm_head_full_k4_s0
  f0_local_how2sign+rand_head-lm_head_full_k4_s0
  f0_local_how2sign+how2sign_cjk-lm_head_full_k4_s0
  f0_local_how2sign+how2sign_cjkgl-lm_head_full_k4_s0
  f0_local_how2sign+how2sign_gl-lm_head_full_k4_s0
  f0_local_openasl+mt5_base-lm_head_full_k4_s0
  f0_local_wlasl+mt5_base-lm_head_full_k4_s0
  f0_local_openasl+csl_daily-lm_head_full_k4_s0
  f0_local_wlasl+csl_daily-lm_head_full_k4_s0
)

DONE=(); FAILED=(); SKIPPED=()
echo "### rc5q starts $(date '+%F %H:%M'), ${#CELLS[@]} cells -> $OUT_NAME"
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
echo "### rc5q complete $(date '+%F %H:%M'): ${#DONE[@]} scored, ${#FAILED[@]} failed, ${#SKIPPED[@]} skipped"
[ ${#FAILED[@]}  -ne 0 ] && printf 'failed:  %s\n'  "${FAILED[@]}"
[ ${#SKIPPED[@]} -ne 0 ] && printf 'skipped: %s\n' "${SKIPPED[@]}"
exit 0
