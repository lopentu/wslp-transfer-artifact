#!/usr/bin/env bash
# Free-decoding BLEU/chrF for the transfer table — limitation L1.
#
# The paper's entire argument rests on 4-way contrastive pronoun accuracy, which
# is a referential diagnostic and not translation quality. A reviewer will ask
# whether the initialisation effect shows up as a translation result at all, and
# right now we cannot answer. This closes that.
#
# Scope, deliberately: the six initialisations on **fold 0 only**. Beam search
# over the 1,761-utterance test split is the expensive half of evaluation, and
# eighteen cells of it would cost more GPU than the entire training sweep. One
# fold gives a BLEU column for every row of Table 1, which is what L1 needs; it
# does not give per-fold BLEU variance, and the paper must not imply that it
# does.
#
# `run_all.sh gen` is the wrong tool here — it loops over *every* run on disk,
# which is now ~40 checkpoints including split and cross-paired rows whose BLEU
# nobody will read.
#
# Two things to know before reading the output:
#
#   1. **Report both BLEU columns.** The CSL checkpoints were pretrained to emit
#      simplified Chinese; this corpus's targets are traditional. On sacrebleu's
#      `zh` tokeniser a correct sentence in the wrong script scores 14 against
#      the identical traditional string's 100. `tsl.metrics.corpus_bleu_chrf`
#      therefore reports raw and script-normalised figures, and the normalised
#      one needs `opencc`:
#
#          .venv/bin/pip install --no-deps opencc-python-reimplemented
#
#      `--no-deps` on purpose: this box shares a venv with a running sweep and a
#      transitive upgrade could change a dependency underneath it.
#
#   2. **Nothing here trains.** Every cell re-reads an existing `best.pt`, so a
#      cell killed by the deadline costs only its own inference and can be
#      re-run tomorrow with no loss.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

INITS=${INITS:-csl_daily csl_stage1 random how2sign openasl wlasl}
FOLD=${FOLD:-0}
TUNE=${TUNE:-full}
CTXK=${CTXK:-4}
SEED=${SEED:-0}

# Inference needs far less than training's 14.4 GB, so this can take a window
# that a training cell could not.
GPU_NEED_MIB=${GPU_NEED_MIB:-8000}
PLAN_BY=${PLAN_BY:-20:00}
HARD_STOP=${HARD_STOP:-20:45}
CELL_BUDGET_MIN=${CELL_BUDGET_MIN:-20}

PLAN_TS=$(date -d "today $PLAN_BY" +%s) || exit 1
HARD_TS=$(date -d "today $HARD_STOP" +%s) || exit 1
secs_to_plan() { echo $(( PLAN_TS - $(date +%s) )); }
secs_to_hard() { echo $(( HARD_TS - $(date +%s) )); }
have_time() { [ "$(secs_to_plan)" -ge $(( CELL_BUDGET_MIN * 60 )) ]; }

wait_for_gpu() {
  local waited=0 free
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -ge "$GPU_NEED_MIB" ] && return 0
    have_time || { echo "### plan-by reached waiting for the GPU"; return 1; }
    [ $((waited % 900)) -eq 0 ] && echo "### waiting for the GPU: ${free} MiB free  ($(date +%H:%M))"
    sleep 60; waited=$((waited + 60))
  done
}

# Do not race the training sweep: it needs 16 GB and we would take the window
# out from under it.
#
# Two ways to get this wrong, and both have now been hit:
#
#   1. Pattern too broad -> the waiter matches *itself* and polls forever. Cost
#      3.5 h of idle card on 8/13.
#   2. Pattern that cannot match -> no wait at all. `pgrep -f` takes an ERE, in
#      which `\?` is a literal question mark, so the "fix" for (1),
#      `run_mechanism_sweep2\?\.sh`, matched nothing and this script started on
#      top of a running training cell on 8/14.
#
# So: match the training entry points *and* the trainer itself, then assert the
# pattern does not match this very process before trusting it.
BUSY_RE='run_mechanism_sweep[0-9]*\.sh|run_pose_sweep\.sh|run_all\.sh|scripts/04_train\.py'
if pgrep -f "$BUSY_RE" 2>/dev/null | grep -qx "$$"; then
  echo "### BUG: the wait pattern matches this process ($$); refusing to start"
  exit 3
fi
while pgrep -f "$BUSY_RE" >/dev/null 2>&1; do
  have_time || { echo "### plan-by reached while waiting for the training sweep"; exit 2; }
  echo "### waiting for the training sweep to finish  ($(date +%H:%M))"
  sleep 120
done

FAILED=(); DEFERRED=(); DONE=0
echo "### BLEU sweep: ${INITS} on fold ${FOLD}  ($(date +%H:%M))"
for init in $INITS; do
  name="f${FOLD}_local_${init}_${TUNE}_k${CTXK}_s${SEED}"
  if [ ! -f "runs/$name/best.pt" ]; then
    echo "--- $name has no checkpoint, skipping"; continue
  fi
  if [ -f "runs/$name/eval_gen.json" ]; then
    echo "--- $name already decoded"; DONE=$((DONE + 1)); continue
  fi
  have_time || { DEFERRED+=("$name"); continue; }
  wait_for_gpu || { DEFERRED+=("$name"); continue; }
  echo "--- $name  ($(date +%H:%M), $(( $(secs_to_plan) / 60 )) min to plan-by)"
  set +e
  timeout -k 60 "$(secs_to_hard)" python3 scripts/05_eval.py \
    --ckpt "runs/$name/best.pt" --conditions local \
    --generate --gen-conditions local --out "runs/$name/eval_gen.json"
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then FAILED+=("gen $name"); else DONE=$((DONE + 1)); fi
done

echo "### $DONE decoded  ($(date +%H:%M))"
[ ${#DEFERRED[@]} -ne 0 ] && { echo "### ${#DEFERRED[@]} deferred (out of time, not failures):"; printf '  %s\n' "${DEFERRED[@]}"; }
[ ${#FAILED[@]} -ne 0 ] && { echo "### ${#FAILED[@]} failed:"; printf '  %s\n' "${FAILED[@]}"; exit 1; }
exit 0
