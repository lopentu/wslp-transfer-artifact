#!/usr/bin/env bash
# Sweep-up: the cells that were gated on another queue's output, plus the two
# stragglers the first pass could not schedule.
#
#   1  The stronger text-only Chinese instrument (ReviewerG's closing question,
#      ReviewerA's "weak instrument"). Appendix O's `tsl_text` head moves 0.026 in
#      relative L2 where every sign fine-tune moves 0.92-0.98, so its null carries
#      no information. ReviewerG's suggestion is a much larger Chinese corpus;
#      there isn't one on this box, and the README's own diagnosis is that the
#      limit is the OBJECTIVE, not the data. So push the same objective until the
#      projection has actually travelled: a learning-rate/epoch ladder, stopping
#      at whichever rung first reaches sign-fine-tune drift, and then the
#      downstream cell. If a Chinese-adapted projection that moved as far as
#      How2Sign's still behaves like mT5-base's, "de-specialization, not
#      written-language match" is established on a matched instrument.
#
#   2  run_rc5.sh's blocks 6 -- `how2sign + gloss_lm lm_head` and
#      `csl_daily-pose + gloss_lm lm_head` -- which SKIP rather than wait when the
#      donor is not yet on disk. run_rc5f.sh builds that donor, so re-entering the
#      training queue after both have run picks them up; the queue is resumable at
#      cell granularity, so re-entering costs nothing for the cells already done.
#
#   3  run_rc5f.sh's block D, free decoding, for the cells that had not finished
#      training when it reached them (the random-head and seeded cells). Same
#      argument: re-entering skips what exists.
#
# Waits for rc5, rc5f and rc5g by exact snapshot name. Not `[r]un_rc5` -- that
# matches this script's own snapshot and it would wait on itself.
#
#   cp run_rc5h.sh .run_rc5h.running.sh && nohup ./.run_rc5h.running.sh \
#       > runs/logs/rc5_sweep.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PLAN_BY=${PLAN_BY:-2026-08-24 06:00}
HARD_STOP=${HARD_STOP:-2026-08-24 08:00}
GPU_TRAIN_MIB=${GPU_TRAIN_MIB:-11000}
HOLD_FILE=${HOLD_FILE:-.gpu-hold}
# The drift a sign fine-tune produces in `lm_head`, from data/head_geometry.json:
# ASL 0.92-0.94, CSL 0.96-0.98. The ladder stops at the first rung that clears
# this, so the instrument is matched rather than merely "stronger".
DRIFT_TARGET=${DRIFT_TARGET:-0.85}

PLAN_TS=$(date -d "$PLAN_BY" +%s) || exit 1
HARD_TS=$(date -d "$HARD_STOP" +%s) || exit 1
secs_to_hard() { echo $(( HARD_TS - $(date +%s) )); }
have_time() { [ "$(( PLAN_TS - $(date +%s) ))" -ge 900 ]; }
gpu_free()  { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

wait_gpu() {
  local need=${1:-$GPU_TRAIN_MIB} waited=0
  while :; do
    [ -f "$HOLD_FILE" ] || { [ "$(gpu_free)" -ge "$need" ] && return 0; }
    have_time || return 1
    [ $(( waited % 900 )) -eq 0 ] && echo "### waiting for ${need} MiB ($(date +%H:%M))"
    sleep 60; waited=$(( waited + 60 ))
  done
}

for pat in "[r]un_rc5f[.]running" "[r]un_rc5g[.]running" "[r]un_rc5[.]running"; do
  while pgrep -f "$pat" > /dev/null; do
    have_time || { echo "### out of time waiting for $pat"; exit 1; }
    echo "### waiting for $pat ($(date +%H:%M))"
    sleep 300
  done
done

# ------------------------------------------------------------------------- (1)
STRONG=data/external/tsl_text_lm_strong.pth
if [ ! -f "$STRONG" ]; then
  # lr | epochs. Ascending, and the loop stops at the first rung whose drift
  # clears DRIFT_TARGET -- the point is a MATCHED instrument, and overshooting
  # it by an order of magnitude would just be a differently unmatched one.
  for rung in "3e-4|24" "1e-3|24" "3e-3|24"; do
    IFS='|' read -r lr ep <<< "$rung"
    rep="data/text_lm_strong_lr${lr}.json"
    if [ ! -f "$rep" ]; then
      wait_gpu || break
      echo "--- tsl_text_strong lr=$lr epochs=$ep ($(date +%H:%M))"
      timeout -k 60 "$(secs_to_hard)" "$PY" scripts/23_text_lm.py \
        --fold 0 --lr "$lr" --epochs "$ep" \
        --out "data/external/tsl_text_lm_strong_lr${lr}.pth" --report "$rep" \
        2>&1 | grep -v "^Loading weights"
      [ -f "$rep" ] || { echo "### rung $lr failed"; continue; }
    fi
    drift=$("$PY" -c "
import json;print(json.load(open('$rep'))['lm_head_rel_l2_from_mt5base'])")
    echo "### lr=$lr drift=$drift (target >= $DRIFT_TARGET)"
    ok=$("$PY" -c "print(1 if float('$drift') >= float('$DRIFT_TARGET') else 0)")
    if [ "$ok" = "1" ]; then
      cp "data/external/tsl_text_lm_strong_lr${lr}.pth" "$STRONG"
      echo "### matched instrument: lr=$lr, drift $drift -> $STRONG"
      break
    fi
  done
  # If no rung cleared the target, take the furthest one and let the report say
  # so: a partly-matched instrument with its drift stated beats a silent gap.
  if [ ! -f "$STRONG" ]; then
    best=$("$PY" - <<'EOF'
import glob, json
rows = [(json.load(open(p))["lm_head_rel_l2_from_mt5base"], p) for p in
        glob.glob("data/text_lm_strong_lr*.json")]
print(max(rows)[1] if rows else "")
EOF
)
    if [ -n "$best" ]; then
      lr=$(echo "$best" | sed 's|.*_lr\(.*\)\.json|\1|')
      cp "data/external/tsl_text_lm_strong_lr${lr}.pth" "$STRONG"
      echo "### no rung reached $DRIFT_TARGET; using the furthest (lr=$lr)"
    fi
  fi
fi

if [ -f "$STRONG" ]; then
  name=f0_local_how2sign+tsl_text_strong-lm_head_full_k4_s0
  if [ ! -f "runs/$name/best.pt" ]; then
    wait_gpu 16000 && {
      echo "--- $name ($(date +%H:%M))"
      timeout -k 60 "$(secs_to_hard)" "$PY" scripts/04_train.py \
        --fold 0 --system local --init how2sign --init-parts all \
        --init-mt5 tsl_text_strong --init-mt5-parts lm_head \
        --tune full --ctx-k 4 --epochs 12 --seed 0
    }
  fi
  for spec in "eval.json|data|local,text_only" \
              "eval_lex.json|data/lex|local,blank_plain" \
              "eval_clip.json|data/lex|local,blank_plain,swap_plain@0,swap_plain@1,swap_plain@2,shuffle_frames"; do
    IFS='|' read -r out data conds <<< "$spec"
    [ -f "runs/$name/$out" ] && continue
    [ -f "runs/$name/best.pt" ] || continue
    wait_gpu 7000 || break
    timeout -k 60 "$(secs_to_hard)" "$PY" scripts/05_eval.py \
      --ckpt "runs/$name/best.pt" --data "$data" --batch 4 --workers 2 \
      --conditions "$conds" --out "runs/$name/$out" 2>&1 | grep -v "^Loading weights"
  done
fi

# ---------------------------------------------------------------------- (2)(3)
echo "### re-entering the training queue for the donor-gated cells ($(date +%H:%M))"
cp run_rc5.sh .run_rc5b.running.sh && chmod +x .run_rc5b.running.sh
PLAN_BY="$PLAN_BY" HARD_STOP="$HARD_STOP" ./.run_rc5b.running.sh || true

echo "### re-entering the forward queue for the remaining decodes ($(date +%H:%M))"
cp run_rc5f.sh .run_rc5fb.running.sh && chmod +x .run_rc5fb.running.sh
PLAN_BY="$PLAN_BY" HARD_STOP="$HARD_STOP" WAIT_FWD="" ./.run_rc5fb.running.sh || true

echo "### rc5h complete ($(date '+%F %H:%M'))"
