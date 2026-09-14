#!/usr/bin/env bash
# The follow-ups the v4 queue could not carry, because two of them are answers to
# what the queue's own first block found.
#
# 1. The CSL-side compatible-head control. Block P showed that grafting
#    How2Sign's fine-tuned lm_head onto a fine-tuned CSL-Daily model costs it 14
#    points. That number is uninterpretable on its own: two independently trained
#    models do not share a coordinate system, so replacing 192.1M parameters with
#    another run's version of them costs something whatever the written language
#    is. Seed 1's own head is that cost with the language held fixed. The
#    How2Sign side already has its version (eval_posthoc_selfseed.json); this is
#    the CSL side, and without it the asymmetry between the two directions cannot
#    be quoted.
#
# 2. Where the wrong clip moves the network (scripts/25_encoder_shift.py). Block
#    P measures clip dependence at the output; this measures it at the mT5
#    encoder as well, which is the depth that separates "the video is not read"
#    from "the video is read and the readout cannot express it".
#
# Runs after the main queue, and waits for it rather than racing it. `[r]un_rc4`
# and not `run_rc4`: an unbracketed pattern matches this script's own command
# line and the wait never ends.
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
PLAN_BY=${PLAN_BY:-2026-08-22 19:30}
HARD_STOP=${HARD_STOP:-2026-08-22 20:00}
GPU_EVAL_MIB=${GPU_EVAL_MIB:-7000}
POST_CONDS=${POST_CONDS:-local,swap_plain@0,swap_plain@1,swap_plain@2}
# NOT `[r]un_rc4`. The bracket trick stops a pattern matching the pgrep process
# itself, and does nothing about a pattern that matches SIBLING scripts sharing a
# name prefix --- including this one. On 2026-08-22 `[r]un_rc4` matched
# `.run_rc4b.running.sh`, so this script waited on itself and sat through the main
# queue's completion. The pattern must name the queue's own snapshot exactly.
WAIT_FOR=${WAIT_FOR:-[r]un_rc4[.]running}

PLAN_TS=$(date -d "$PLAN_BY" +%s) || exit 1
HARD_TS=$(date -d "$HARD_STOP" +%s) || exit 1
secs_to_hard() { echo $(( HARD_TS - $(date +%s) )); }
have_time() { [ "$(( PLAN_TS - $(date +%s) ))" -ge 300 ]; }
gpu_free()  { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

if [ -n "$WAIT_FOR" ]; then
  while pgrep -f "$WAIT_FOR" > /dev/null; do
    echo "### waiting for the main queue  ($(date +%H:%M))"
    sleep 300
  done
fi

wait_gpu() {
  local waited=0
  while :; do
    [ "$(gpu_free)" -ge "$GPU_EVAL_MIB" ] && return 0
    have_time || return 1
    [ $(( waited % 900 )) -eq 0 ] && echo "### waiting for the card ($(date +%H:%M))"
    sleep 60; waited=$(( waited + 60 ))
  done
}

FAILED=()

POSTHOC=(
  "f0_local_csl_daily_full_k4_s0|f0_local_csl_daily_full_k4_s1|eval_posthoc_selfseed.json"
  "f1_local_how2sign_full_k4_s0|f1_local_how2sign+csl_daily-lm_head_full_k4_s0|eval_posthoc_rescued.json"
  "f2_local_how2sign_full_k4_s0|f2_local_how2sign+csl_daily-lm_head_full_k4_s0|eval_posthoc_rescued.json"
)
for spec in "${POSTHOC[@]}"; do
  IFS='|' read -r rcp donor out <<< "$spec"
  [ -f "runs/$rcp/$out" ] && { echo "--- $out on $rcp already done"; continue; }
  [ -f "runs/$donor/best.pt" ] || { echo "### donor $donor missing"; FAILED+=("$donor"); continue; }
  wait_gpu || { echo "### out of time before $out"; break; }
  echo "--- posthoc $rcp <- $donor  ($(date +%H:%M))"
  timeout -k 60 "$(secs_to_hard)" "$PY" scripts/05_eval.py \
    --ckpt "runs/$rcp/best.pt" --data data/lex --batch 4 --workers 2 \
    --conditions "$POST_CONDS" --posthoc-head "runs/$donor/best.pt" \
    --out "runs/$rcp/$out" 2>&1 | grep -v "^Loading weights"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { rm -f "runs/$rcp/$out"; FAILED+=("$out $rcp"); }
done

if [ ! -f data/encoder_shift.json ]; then
  wait_gpu && {
    echo "--- encoder shift  ($(date +%H:%M))"
    timeout -k 60 "$(secs_to_hard)" "$PY" scripts/25_encoder_shift.py --within \
      --runs f0_local_csl_daily_full_k4_s0 f0_local_how2sign_full_k4_s0 \
             f0_local_random_full_k4_s0 \
             f0_local_how2sign+csl_daily-lm_head_full_k4_s0 \
      2>&1 | grep -v "^Loading weights"
    [ "${PIPESTATUS[0]}" -eq 0 ] || FAILED+=("encoder_shift")
  }
fi

echo
if [ ${#FAILED[@]} -ne 0 ]; then
  echo "### ${#FAILED[@]} failed:"; printf '  %s\n' "${FAILED[@]}"; exit 1
fi
echo "### rc4b complete  ($(date +%H:%M))"
