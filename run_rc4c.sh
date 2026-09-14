#!/usr/bin/env bash
# Folds 1 and 2 for the cell that turned out to decide the paper's framing.
#
# Block T's first cell answered ReviewerA's A1 and answered it against the paper:
# `how2sign + mt5_base lm_head` scores 47.9% on fold 0 against 47.6% for
# `how2sign + csl_daily lm_head` and 24.5% intact. So the whole of the effect
# S3.3 localizes is the REMOVAL of an English-specialized output projection, and
# adding a Chinese one on top of that is worth nothing measurable. That is the
# reading the review said would not weaken the paper but would change its
# framing, and it is right on both counts -- which makes this cell load-bearing,
# and a load-bearing cell may not be a single 330-item fold when every other row
# in Table 1 is pooled over 940.
#
# Two cells, ~30 min each. Queued behind both earlier queues rather than beside
# them; `[r]un_rc4` matches both without matching this script's own command line.
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
PLAN_BY=${PLAN_BY:-2026-08-22 19:30}
HARD_STOP=${HARD_STOP:-2026-08-22 20:00}
CELL_BUDGET_MIN=${CELL_BUDGET_MIN:-32}
GPU_NEED_MIB=${GPU_NEED_MIB:-16000}
GPU_EVAL_MIB=${GPU_EVAL_MIB:-7000}
HOLD_FILE=${HOLD_FILE:-.gpu-hold}
CLIP_CONDS=${CLIP_CONDS:-local,blank_plain,swap_plain@0,swap_plain@1,swap_plain@2,shuffle_frames}
# Exact snapshot name, for the reason in run_rc4b.sh: a prefix pattern can match
# this script's own command line and wait on itself forever.
WAIT_FOR=${WAIT_FOR:-[r]un_rc4b[.]running}

PLAN_TS=$(date -d "$PLAN_BY" +%s) || exit 1
HARD_TS=$(date -d "$HARD_STOP" +%s) || exit 1
secs_to_hard() { echo $(( HARD_TS - $(date +%s) )); }
have_time()      { [ "$(( PLAN_TS - $(date +%s) ))" -ge $(( CELL_BUDGET_MIN * 60 )) ]; }
have_time_eval() { [ "$(( PLAN_TS - $(date +%s) ))" -ge 300 ]; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

wait_gpu() {
  local need=${1:-$GPU_NEED_MIB} waited=0
  while :; do
    [ -f "$HOLD_FILE" ] || { [ "$(gpu_free)" -ge "$need" ] && return 0; }
    have_time || return 1
    [ $(( waited % 900 )) -eq 0 ] && echo "### waiting for ${need} MiB ($(date +%H:%M))"
    sleep 60; waited=$(( waited + 60 ))
  done
}

# Guarded: an empty WAIT_FOR would make `pgrep -f ""` match every process on the
# machine and hold this queue forever.
while [ -n "$WAIT_FOR" ] && pgrep -f "$WAIT_FOR" > /dev/null; do
  echo "### waiting for the earlier queues ($(date +%H:%M))"
  sleep 300
done

CELLS=(
  "1|how2sign|all|mt5_base|lm_head|0|12"
  "2|how2sign|all|mt5_base|lm_head|0|12"
  # Lower priority: the CSL-side neutral cell, which is a symmetry check on a
  # result the ASL side already carries.
  "1|csl_daily|all|mt5_base|lm_head|0|12"
  "2|csl_daily|all|mt5_base|lm_head|0|12"
)
FAILED=(); DEFERRED=()

cell_done() {
  local name=$1 got
  [ -f "runs/$name/best.pt" ] && [ -f "runs/$name/log.json" ] || return 1
  got=$("$PY" -c "import json;print(json.load(open('runs/$name/log.json'))['epochs_done'])" 2>/dev/null) || return 1
  [ "${got:-0}" -ge 12 ]
}

run_eval() {
  local name=$1 out=$2 data=$3 conds=$4
  [ -f "runs/$name/$out" ] && return 0
  have_time_eval || { DEFERRED+=("eval $out $name"); return 1; }
  wait_gpu "$GPU_EVAL_MIB" || { DEFERRED+=("eval $out $name"); return 1; }
  timeout -k 60 "$(secs_to_hard)" "$PY" scripts/05_eval.py \
    --ckpt "runs/$name/best.pt" --data "$data" --batch 4 --workers 2 \
    --conditions "$conds" --out "runs/$name/$out" 2>&1 | grep -v "^Loading weights"
  [ "${PIPESTATUS[0]}" -eq 0 ] && [ -f "runs/$name/$out" ] && return 0
  rm -f "runs/$name/$out"; FAILED+=("eval $out $name"); return 1
}

for spec in "${CELLS[@]}"; do
  IFS='|' read -r f init parts mt5 mt5parts seed epochs <<< "$spec"
  name="f${f}_local_${init}+${mt5}-${mt5parts}_full_k4_s${seed}"
  if cell_done "$name"; then
    echo "--- $name already trained"
  else
    have_time || { DEFERRED+=("$name"); continue; }
    wait_gpu  || { DEFERRED+=("$name"); continue; }
    echo "--- $name ($(date +%H:%M))"
    timeout -k 60 "$(secs_to_hard)" "$PY" scripts/04_train.py \
      --fold "$f" --system local --init "$init" --init-parts "$parts" \
      --init-mt5 "$mt5" --init-mt5-parts "$mt5parts" \
      --tune full --ctx-k 4 --epochs "$epochs" --seed "$seed"
    rc=$?
    if [ "$rc" -ne 0 ]; then FAILED+=("train $name (exit $rc)"); continue; fi
  fi
  run_eval "$name" eval.json       data     local,text_only   || continue
  run_eval "$name" eval_blank.json data     blank_plain       || continue
  run_eval "$name" eval_lex.json   data/lex local,blank_plain || continue
  run_eval "$name" eval_clip.json  data/lex "$CLIP_CONDS"     || continue
done

echo
[ ${#DEFERRED[@]} -ne 0 ] && { echo "### deferred to tomorrow:"; printf '  %s\n' "${DEFERRED[@]}"; }
[ ${#FAILED[@]} -ne 0 ] && { echo "### failed:"; printf '  %s\n' "${FAILED[@]}"; exit 1; }
echo "### rc4c complete ($(date +%H:%M))"
