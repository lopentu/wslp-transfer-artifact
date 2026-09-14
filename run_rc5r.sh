#!/usr/bin/env bash
# The referential wrong-clip reading on folds 1 and 2, for every cell that has a
# three-fold `eval.json` and a one-fold `eval_clip_pron.json`.
#
# This closes the last documented gap in numbers.tex and it is a real one, not a
# tidy-up. Right now every `HeadClip*Pron` number is fold 0, 313 items, while its
# `Lex` twin is three folds, 940 items. Both are valid within-cell paired
# contrasts, but they do not share a denominator, so `make_numbers.py` refuses to
# print the 30 `*Pron{Swap,Shuffle}*` macros at all -- the guard fires with
# "clip file covers fewer folds than eval" -- and every Lex/Pron table in the
# report has to carry a sentence saying which is which. Scoring folds 1 and 2
# makes the two families the same shape and removes both problems.
#
# The guide put this at "~2 h of card time, worth knowing about, not obviously
# worth doing". That estimate predates a measurement: rc5p and rc5q ran these
# evals at 2m09s-2m41s each, not the 4 min the estimate assumed. 26 cells at
# ~2.5 min is ~65 min, which changes the answer.
#
# SEQUENCING. rc5q is still running and both queues want the same card, so this
# gates on rc5q's PID before it starts. 2565755 is the queue shell itself
# (ppid 1, the only match for `.run_rc5q.running.sh`, no wrapper quoting the
# name) -- which is the trap that cost run_rc5g.sh and run_rc5h.sh their whole
# run, so it is worth stating that it was checked rather than assumed.
#
# GPU. Same arrangement Kevin authorised for rc5p and rc5q: `.gpu-hold` bypassed
# because richard's `ollama serve` is a persistent model server and the hold will
# not clear on its own, GPU_NEED_MIB raised to 13000 against a measured 3.3 GB of
# real use, re-checked before every cell so a growing neighbour parks the rest of
# the queue instead of colliding with it. `.gpu-hold` is left in place.
#
#   cp run_rc5r.sh .run_rc5r.running.sh && chmod +x .run_rc5r.running.sh
#   WAIT_PID=2565755 HOLD_FILE=.gpu-hold.bypassed-for-rc5r GPU_NEED_MIB=13000 \
#     nohup ./.run_rc5r.running.sh >> runs/logs/rc5_r.log 2>&1 &
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
WAIT_PID=${WAIT_PID:-}
PLAN_BY=${PLAN_BY:-2026-08-24 06:00}
HARD_STOP=${HARD_STOP:-2026-08-24 08:00}

PLAN_TS=$(date -d "$PLAN_BY" +%s) || exit 1
HARD_TS=$(date -d "$HARD_STOP" +%s) || exit 1
secs_to_hard() { echo $(( HARD_TS - $(date +%s) )); }
have_time() { [ "$(( PLAN_TS - $(date +%s) ))" -ge 600 ]; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

# Wait for the queue that is already on the card. A PID, not a name.
if [ -n "$WAIT_PID" ]; then
  waited=0
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    have_time || { echo "### out of time waiting on pid $WAIT_PID"; exit 1; }
    [ $(( waited % 600 )) -eq 0 ] && echo "### waiting for queue pid $WAIT_PID ($(date '+%m-%d %H:%M'))"
    sleep 60; waited=$(( waited + 60 ))
  done
  echo "### queue pid $WAIT_PID has exited ($(date '+%m-%d %H:%M'))"
fi
# Belt and braces: no eval of any provenance should be mid-flight.
while pgrep -f "[0]5_eval[.]py" > /dev/null; do sleep 30; done

wait_for_gpu() {
  local waited=0
  while :; do
    [ -f "$HOLD_FILE" ] || { [ "$(gpu_free)" -ge "$GPU_NEED_MIB" ] && return 0; }
    have_time || return 1
    [ $(( waited % 900 )) -eq 0 ] && echo "### waiting for ${GPU_NEED_MIB} MiB ($(date '+%m-%d %H:%M'))"
    sleep 60; waited=$(( waited + 60 ))
  done
}

# Built by rule, not typed: every fold-1/fold-2 cell that has an `eval.json` and
# no `eval_clip_pron.json`. Typing 26 run names by hand is how one gets missed.
mapfile -t CELLS < <(
  for d in runs/f[12]_local_*_full_k4_s0; do
    [ -f "$d/eval.json" ] || continue
    [ -f "$d/$OUT_NAME" ] && continue
    [ -f "$d/best.pt" ] || continue
    basename "$d"
  done
)

DONE=(); FAILED=(); SKIPPED=()
echo "### rc5r starts $(date '+%F %H:%M'), ${#CELLS[@]} cells -> $OUT_NAME"
for n in "${CELLS[@]}"; do
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
echo "### rc5r complete $(date '+%F %H:%M'): ${#DONE[@]} scored, ${#FAILED[@]} failed, ${#SKIPPED[@]} skipped"
[ ${#FAILED[@]}  -ne 0 ] && printf 'failed:  %s\n'  "${FAILED[@]}"
[ ${#SKIPPED[@]} -ne 0 ] && printf 'skipped: %s\n' "${SKIPPED[@]}"
exit 0
