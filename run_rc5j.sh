#!/usr/bin/env bash
# Free decoding for the cells rc5f could not reach, and for the one cell the
# 8/23 seed analysis turned into a headline.
#
#   A  the abolition cell, 3 seeds. `csl_daily + how2sign lm_head` at
#      INITIALIZATION, then 12 epochs of full fine-tuning, lands at 24.0% --
#      chance -- with Delta local-swap of exactly 0.00 [-1.61, +1.61], p = 1.0,
#      down from CSL-Daily's own +20.9. Nothing else in the table abolishes
#      grounding; the wrong-head grafts merely fail to create it. Teacher-forced
#      scoring cannot say whether the text is degenerate or merely wrong, and
#      that distinction is the whole content of section 6, so it needs beam
#      search. Three seeds because the accuracy drop replicated at three
#      (-30.6 / -31.8 / -30.3) and a one-seed generation claim next to a
#      three-seed scoring claim invites exactly the O-P1 objection again.
#
#   B  the seed replication of the two neutral-head cells' generation. Section 6
#      currently rests on fold-0 seed-0 for the degeneracy claim and on folds
#      1/2 for its replication; seeds 1/2 of fold 0 close the other axis.
#
#   C  the two rand_head_nm cells rc5f deferred because they had no best.pt yet.
#      Gated on 04_train.py being absent, NOT on a script filename -- the
#      launcher shell quotes the snapshot's name, so a filename gate matches the
#      launcher and never clears.
#
#   cp run_rc5j.sh .run_rc5j.running.sh && nohup ./.run_rc5j.running.sh \
#       > runs/logs/rc5_j.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PLAN_BY=${PLAN_BY:-2026-08-24 06:00}
HARD_STOP=${HARD_STOP:-2026-08-24 08:00}
GPU_EVAL_MIB=${GPU_EVAL_MIB:-8000}
HOLD_FILE=${HOLD_FILE:-.gpu-hold}

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

# "is a training cell running right now" -- the process, not a filename.
train_running() { pgrep -f "[0]4_train[.]py" >/dev/null 2>&1; }
wait_no_train() {
  local waited=0
  while train_running; do
    have_time || return 1
    [ $(( waited % 1800 )) -eq 0 ] && echo "### waiting for training to finish ($(date +%H:%M))"
    sleep 120; waited=$(( waited + 120 ))
  done
  return 0
}

FAILED=(); DEFERRED=()

# `best.pt` is rewritten EVERY epoch, so its existence does not mean the cell
# finished; `log.json`'s `epochs_done` does. Copied from run_wrongclip.sh's
# `cell_done` after a near miss on 8/23: `wait_no_train` cleared during the gap
# BETWEEN two training cells -- a queue sitting in `wait_gpu` has no python child
# and reads as idle -- and rc5j went on to score `csl_daily+rand_head_nm`.
# It happened to have finished four minutes earlier (12/12 epochs, best.pt at
# 00:51:46, eval at 00:59:32), so the number is good; had the gap fallen the
# other side of that cell the eval would have scored a half-trained model and
# nothing in the output would have said so.
cell_done() {
  local n=$1 want=12 got
  case "$n" in *_e24) want=24;; *_e20) want=20;; esac
  [ -f "runs/$n/best.pt" ] && [ -f "runs/$n/log.json" ] || return 1
  got=$("$PY" -c "import json;print(json.load(open('runs/$n/log.json'))['epochs_done'])" 2>/dev/null) || return 1
  [ "${got:-0}" -ge "$want" ]
}

run_gen() {  # run  outfile
  local run=$1 out=${2:-eval_gen.json}
  [ -f "runs/$run/$out" ] && { echo "--- gen $out on $run already done"; return 0; }
  cell_done "$run" || { echo "--- gen $run not finished training, skipping"; DEFERRED+=("gen $run"); return 0; }
  wait_gpu || { DEFERRED+=("gen $run"); return 1; }
  echo "--- gen $run ($(date +%H:%M))"
  timeout -k 60 "$(secs_to_hard)" "$PY" scripts/05_eval.py \
    --ckpt "runs/$run/best.pt" --data data --batch 4 --workers 2 \
    --conditions local --generate --gen-conditions local \
    --out "runs/$run/$out" 2>&1 | grep -v "^Loading weights"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { rm -f "runs/$run/$out"; FAILED+=("gen $run"); }
}

# ---------------------------------------------------------------------- block A
for s in 0 1 2; do
  run_gen "f0_local_csl_daily+how2sign-lm_head_full_k4_s${s}" || break
done

# ---------------------------------------------------------------------- block B
for s in 1 2; do
  run_gen "f0_local_how2sign+mt5_base-lm_head_full_k4_s${s}" || break
  run_gen "f0_local_csl_daily+mt5_base-lm_head_full_k4_s${s}" || break
done

# ---------------------------------------------------------------------- block C
# These two were still training when rc5f reached them, so their best.pt was
# being rewritten; evaluating it then would have scored a half-finished model.
if wait_no_train; then
  run_gen "f0_local_how2sign+rand_head_nm-lm_head_full_k4_s0"
  run_gen "f0_local_csl_daily+rand_head_nm-lm_head_full_k4_s0"
else
  DEFERRED+=("rand_head_nm gens (out of time)")
fi

echo
[ ${#DEFERRED[@]} -ne 0 ] && { echo "### deferred:"; printf '  %s\n' "${DEFERRED[@]}"; }
[ ${#FAILED[@]}   -ne 0 ] && { echo "### failed:"; printf '  %s\n' "${FAILED[@]}"; exit 1; }
echo "### rc5j complete ($(date '+%F %H:%M'))"
