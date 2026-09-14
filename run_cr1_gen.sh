#!/usr/bin/env bash
# Free-decoding pass for the camera-ready permutation cells.
#
# `run_cr1.sh` runs the five contrastive readouts on every cell it trains, but
# not `eval_gen.json` -- unconstrained decoding is a separate, slower pass that
# the submitted paper only ever ran on fold 0 of a handful of cells. The
# camera-ready's Appendix free-decoding table now has a row for the permuted
# projection, so those three macros (\HeadHowPerm{Bleu,Uniq,Modal}) need this.
#
# Waits on the training queue's PID rather than on a script name: the launcher
# shell carries the script's name on its own command line, so a pgrep for it
# matches the wrong process (run_cr1.sh's own guard learned this the hard way).
# Pass the PID in, or let it discover the queue once at startup.
#
#   QUEUE_PID=936238 nohup ./run_cr1_gen.sh >> runs/logs/cr1_gen.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
GPU_NEED_MIB=${GPU_NEED_MIB:-9000}
HARD_STOP=${HARD_STOP:-2026-09-14 20:00}
CELLS=${CELLS:-"f0_local_how2sign+mt5_base_perm-lm_head_full_k4_s0
f0_local_how2sign+mt5_base_permws-lm_head_full_k4_s0"}

HARD_TS=$(date -d "$HARD_STOP" +%s) || exit 1
secs_to_hard() { echo $(( HARD_TS - $(date +%s) )); }

QUEUE_PID=${QUEUE_PID:-$(ps -eo pid,args | awk '/[.]run_cr1\.running\.sh/ {print $1; exit}')}
if [ -n "${QUEUE_PID:-}" ]; then
  echo "### waiting for the training queue (pid $QUEUE_PID) to finish"
  while kill -0 "$QUEUE_PID" 2>/dev/null; do
    [ "$(secs_to_hard)" -le 0 ] && { echo "### past hard stop while waiting"; exit 2; }
    sleep 60
  done
  echo "### queue finished $(date '+%F %H:%M')"
fi

gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

for n in $CELLS; do
  out="runs/$n/eval_gen.json"
  [ -f "$out" ] && { echo "--- $n already decoded"; continue; }
  [ -f "runs/$n/best.pt" ] || { echo "### $n has no checkpoint, skipping"; continue; }
  waited=0
  while [ "$(gpu_free)" -lt "$GPU_NEED_MIB" ]; do
    [ "$(secs_to_hard)" -le 0 ] && { echo "### past hard stop"; exit 2; }
    sleep 60; waited=$(( waited + 60 ))
    [ $(( waited % 600 )) -eq 0 ] && echo "### waiting for ${GPU_NEED_MIB} MiB"
  done
  echo "--- decoding $n ($(date '+%m-%d %H:%M'))"
  timeout -k 60 "$(secs_to_hard)" "$PY" scripts/05_eval.py \
      --ckpt "runs/$n/best.pt" --conditions local \
      --generate --gen-conditions local --out "$out" 2>&1 | grep -v "^Loading weights"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ] && [ -f "$out" ]; then
    echo "### decoded $n"
  else
    echo "### FAILED gen $n (exit $rc)"; rm -f "$out"
  fi
done
echo "### cr1 generation pass done $(date '+%F %H:%M')"
