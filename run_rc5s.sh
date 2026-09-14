#!/usr/bin/env bash
# The fold-0 half of the job rc5r only did half of.
#
# rc5r's cell list was built by the rule "every f1/f2 cell with an `eval.json` and
# no `eval_clip_pron.json`", on the assumption that fold 0 already had one
# everywhere. It does not. Eighteen fold-0 cells never got a referential
# wrong-clip reading, and for ten of them the *other two folds now do* -- so rc5r
# turned a clean one-fold gap into a ragged two-of-three one, and
# `make_numbers.py`'s coverage guard still refuses the 30 `*Pron{Swap,Shuffle}*`
# macros for exactly those cells. Same guard, opposite fold.
#
# The rule this script uses is the one rc5r should have used: ANY fold whose
# `eval.json` has no `eval_clip_pron.json` beside it. Written as a glob over
# f[012] so it cannot make the same mistake again, and it is a no-op for the 35
# cells already at parity.
#
# Ten of the eighteen unblock macros (the three-fold encoder-side cells:
# {how2sign,openasl,wlasl,csl_daily,csl_stage1}-pose, {csl_daily,csl_stage1}-mt5,
# csl_daily-pose+wlasl-mt5). The other eight are single-fold cells -- A-B6's two
# `_sl` retrains, the label-smoothing cell, the -proj/-noproj ablations -- which
# block nothing but cost 2.5 min each and complete the picture.
#
# GPU: same terms as rc5p/rc5q/rc5r, still needed -- ollama is still resident
# (9,330 MiB, 90% util at 13:02) and `.gpu-hold` is still set.
#
#   cp run_rc5s.sh .run_rc5s.running.sh && chmod +x .run_rc5s.running.sh
#   HOLD_FILE=.gpu-hold.bypassed-for-rc5s GPU_NEED_MIB=13000 \
#     nohup ./.run_rc5s.running.sh >> runs/logs/rc5_s.log 2>&1 &
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
PLAN_BY=${PLAN_BY:-2026-08-24 06:00}
HARD_STOP=${HARD_STOP:-2026-08-24 08:00}

PLAN_TS=$(date -d "$PLAN_BY" +%s) || exit 1
HARD_TS=$(date -d "$HARD_STOP" +%s) || exit 1
secs_to_hard() { echo $(( HARD_TS - $(date +%s) )); }
have_time() { [ "$(( PLAN_TS - $(date +%s) ))" -ge 600 ]; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

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

# ANY fold, not just f1/f2. This is the fix.
mapfile -t CELLS < <(
  for d in runs/f[012]_local_*_full_k4_s0; do
    [ -f "$d/eval.json" ] || continue
    [ -f "$d/$OUT_NAME" ] && continue
    [ -f "$d/best.pt" ] || continue
    basename "$d"
  done
)

DONE=(); FAILED=(); SKIPPED=()
echo "### rc5s starts $(date '+%F %H:%M'), ${#CELLS[@]} cells -> $OUT_NAME"
for n in "${CELLS[@]}"; do
  [ -f "runs/$n/$OUT_NAME" ] && { echo "--- $n already scored"; continue; }
  have_time    || { SKIPPED+=("$n"); continue; }
  wait_for_gpu || { SKIPPED+=("$n"); continue; }
  echo "--- $n ($(date '+%m-%d %H:%M'))"
  timeout -k 60 "$(secs_to_hard)" "$PY" scripts/05_eval.py \
      --ckpt "runs/$n/best.pt" --data "$DATA" --batch "$BATCH" \
      --workers "$WORKERS" --conditions "$CONDITIONS" \
      --out "runs/$n/$OUT_NAME" 2>&1 | grep -v "^Loading weights"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ] && [ -f "runs/$n/$OUT_NAME" ]; then DONE+=("$n")
  else echo "### FAILED $n (exit $rc)"; rm -f "runs/$n/$OUT_NAME"; FAILED+=("$n"); fi
done

echo
echo "### rc5s complete $(date '+%F %H:%M'): ${#DONE[@]} scored, ${#FAILED[@]} failed, ${#SKIPPED[@]} skipped"
[ ${#FAILED[@]}  -ne 0 ] && printf 'failed:  %s\n'  "${FAILED[@]}"
[ ${#SKIPPED[@]} -ne 0 ] && printf 'skipped: %s\n' "${SKIPPED[@]}"
exit 0
