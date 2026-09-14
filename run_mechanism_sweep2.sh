#!/usr/bin/env bash
# What the paper still cannot support, cheapest and most load-bearing first.
#
# The paper's title says the thing that transfers is the encoder-decoder
# *pairing*. The split rows do not actually establish that. They show neither
# half alone suffices, which is equally consistent with a much duller account:
# you simply need a good encoder and a good decoder, and either one alone is
# not enough. Nothing run so far separates the two, and the difference is the
# difference between the paper's most interesting sentence being true and being
# an over-reading of its own evidence. Block 2 below is that experiment.
#
# Cells are listed in priority order and the deadline logic truncates from the
# end, so a sweep that runs out of card leaves the load-bearing rows done and
# the insurance rows undone rather than the other way round.
#
#   fold | init | init-parts | init-mt5 | seed
#
# Block 1  csl_stage1 splits, fold 2. Closes the one gap named in
#          encoder-axis-results.md §6: the CSL-News encoder-only row exists for
#          fold 0 only, and fold 0 is the fold on which two other rows made an
#          excursion, so that row currently cannot be reported at all.
#
# Block 2  Cross-pairing. Pose encoder from CSL-Daily on the decoder from
#          CSL-News, and the reverse. Both halves are then individually
#          pretrained and both speak Chinese; the only thing broken is that
#          they were never trained together.
#            ~78%  -> co-adaptation is wrong, "both halves must be pretrained"
#                     is the mechanism, and the title has to change.
#            ~56%  -> co-adaptation survives a real test for the first time.
#
# Block 3  Where the pairing lives. `pose_proj` is the 0.8 M linear map from
#          the pose branch into the decoder's embedding space -- the only place
#          the two halves touch. If it alone carries a large share of the
#          benefit, the mechanism is an interface rather than a representation,
#          which is a sharper claim than the paper currently makes.
#
# Block 4  Seed variance on the two bimodal rows. How2Sign-pose and WLASL-pose
#          both hit 73.8% on fold 0 and ~55% on folds 1-2. The paper explains
#          that as an optimisation excursion, which is a guess: it has one seed.
#          If seeds 1 and 2 also land at ~74% on fold 0 it is not seed luck at
#          all but something about fold 0, and Table 3's reading is wrong.
#
# Block 5  Seed variance on the headline rows (L7). Defensive, not diagnostic.
#
# Block 6  Seed variance on the encoder-only row the paper reports.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

TUNE=${TUNE:-full}
CTXK=${CTXK:-4}
EPOCHS=${EPOCHS:-12}
EVAL_CONDS=${EVAL_CONDS:-local,text_only}

GPU_NEED_MIB=${GPU_NEED_MIB:-16000}
GPU_WAIT_MAX=${GPU_WAIT_MAX:-43200}

# By agreement (2026-08-12) a colleague has the card 21:00-09:00. PLAN_BY is the
# scheduling limit -- no cell is started that we do not expect to finish by then;
# HARD_STOP is the safety net for a cell that overruns. See run_pose_sweep.sh,
# where the same two clocks are explained at length.
PLAN_BY=${PLAN_BY:-20:00}
HARD_STOP=${HARD_STOP:-20:45}
CELL_BUDGET_MIN=${CELL_BUDGET_MIN:-30}

CELLS=(
  # ---- block 1: close the CSL-News encoder gap
  "2|csl_stage1|pose||0"
  "2|csl_stage1|mt5||0"
  # ---- block 2: cross-pairing, the title's claim
  "0|csl_daily|pose|csl_stage1|0"
  "0|csl_stage1|pose|csl_daily|0"
  # ---- block 3 (NEW, promoted 8/13 15:40): does the encoder's sign language
  # matter at all once the decoder is CSL? Cross-pairing across checkpoints
  # recovered the full benefit (78.6% vs 78.3/79.2 whole), which refutes
  # co-adaptation and makes this the decisive cell: an ASL encoder on a Chinese
  # decoder. ~78% means the encoder's source language is irrelevant and Table 1
  # is entirely about the decoder's output language; ~56% means a genuine
  # sign-language-specific encoder requirement survives. WLASL is the stronger
  # version -- a different pretraining task as well as a different language.
  "0|how2sign|pose|csl_daily|0"
  "0|wlasl|pose|csl_daily|0"
  # The fourth corner. With ASL-enc/ASL-dec already in Table 1 at 45.7 and
  # CSL-enc/CSL-dec at 78.6, these two rows plus this one complete a 2x2 over
  # (encoder sign language) x (decoder written language), so the two factors can
  # be separated by design instead of inferred from three corners.
  "0|csl_daily|pose|how2sign|0"
  # ---- block 4: where the shared interface lives
  "0|csl_daily|proj||0"
  "0|csl_daily|mt5_proj||0"
  "0|csl_daily|pose_noproj||0"
  # ---- block 5: is the fold-0 excursion seed luck?
  "0|how2sign|pose||1"
  "0|wlasl|pose||1"
  "0|how2sign|pose||2"
  "0|wlasl|pose||2"
  # ---- block 6: seed variance on the headline rows
  "0|csl_daily|all||1"
  "0|random|all||1"
  "0|how2sign|all||1"
  "0|csl_daily|all||2"
  "0|random|all||2"
  "0|how2sign|all||2"
  # ---- block 7: seed variance on the reported encoder-only row
  "0|csl_daily|pose||1"
  "0|csl_daily|pose||2"
)

# Hold until the colleague's window (21:00-09:00) is over. Empty = start now.
#
#   START_AFTER=09:00 nohup ./run_mechanism_sweep2.sh &
#
# This exists because the queue was lost once: on 8/13 the requeue was chained
# to a short-lived helper process that did not survive, the card sat idle from
# 15:56 to 19:32, and about eight cells were lost. A sweep that parks itself
# overnight under nohup cannot fail that way -- there is nothing to chain it to.
START_AFTER=${START_AFTER:-}
hold_until() {
  [ -n "$1" ] || return 0
  local target remain
  if ! target=$(date -d "today $1" +%s 2>/dev/null); then
    echo "### START_AFTER=$1 is not a time I can parse; starting now"; return 0
  fi
  if [ "$target" -le "$(date +%s)" ]; then target=$(date -d "tomorrow $1" +%s); fi
  while :; do
    remain=$((target - $(date +%s)))
    [ "$remain" -le 0 ] && break
    echo "### holding until $1 before touching the GPU: $((remain / 60)) min to go  ($(date +%H:%M))"
    if [ "$remain" -lt 1800 ]; then sleep "$remain"; else sleep 1800; fi
  done
  echo "### $1 reached ($(date +%H:%M))"
}

PLAN_TS=""; HARD_TS=""
set_deadlines() {
  [ -n "$PLAN_BY" ] || return 0
  # Computed AFTER the hold, not at launch. Parking overnight and then measuring
  # against a deadline from the previous calendar day would put every deadline
  # in the past and refuse every cell -- which is exactly how a queue that looks
  # scheduled quietly does nothing.
  PLAN_TS=$(date -d "today $PLAN_BY" +%s 2>/dev/null) || {
    echo "### PLAN_BY='$PLAN_BY' unparseable — refusing to run without a deadline"; exit 1; }
  HARD_TS=$(date -d "today $HARD_STOP" +%s 2>/dev/null) || {
    echo "### HARD_STOP='$HARD_STOP' unparseable"; exit 1; }
  if [ "$PLAN_TS" -le "$(date +%s)" ]; then
    echo "### plan-by $PLAN_BY is already past ($(date +%H:%M)) — nothing can run today"; exit 2
  fi
  echo "### plan-by $PLAN_BY (no cell starts with under ${CELL_BUDGET_MIN} min left); hard stop $HARD_STOP"
}

secs_to_plan() { [ -n "$PLAN_TS" ] || { echo 999999; return; }; echo $(( PLAN_TS - $(date +%s) )); }
secs_to_hard() { [ -n "$HARD_TS" ] || { echo 999999; return; }; echo $(( HARD_TS - $(date +%s) )); }
have_time_for_a_cell() {
  [ -n "$PLAN_TS" ] || return 0
  [ "$(secs_to_plan)" -ge $(( CELL_BUDGET_MIN * 60 )) ]
}

# Do not race our own queue for the card: a second job of ours arriving mid-cell
# OOMs one of the two rather than overlapping with it.
wait_for_our_own_queue() {
  local waited=0
  while pgrep -f 'run_all\.sh split|run_pose_sweep\.sh' >/dev/null 2>&1; do
    have_time_for_a_cell || { echo "### plan-by reached while waiting for the other queue"; return 1; }
    if [ $((waited % 900)) -eq 0 ]; then
      echo "### waiting for the earlier queue to finish  ($(date +%H:%M))"
    fi
    sleep 60
    waited=$((waited + 60))
  done
  [ "$waited" -gt 0 ] && echo "### earlier queue done after $((waited / 60)) min"
  return 0
}

wait_for_gpu() {
  local waited=0 free need=${1:-$GPU_NEED_MIB}
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge "$need" ]; then
      [ "$waited" -gt 0 ] && echo "### GPU free (${free} MiB) after $((waited / 60)) min"
      return 0
    fi
    have_time_for_a_cell || { echo "### plan-by reached while waiting for a GPU window"; return 1; }
    if [ "$waited" -ge "$GPU_WAIT_MAX" ]; then
      echo "### gave up: ${free} MiB free, need ${need}, waited $((waited / 60)) min"
      return 1
    fi
    if [ $((waited % 900)) -eq 0 ]; then
      echo "### waiting for the GPU: ${free} MiB free, need ${need} MiB  ($(date +%H:%M))"
    fi
    sleep 60
    waited=$((waited + 60))
  done
}

FAILED=(); DEFERRED=(); DONE=0
run_cell() {
  local f=$1 init=$2 parts=$3 mt5=$4 seed=$5
  # Mirrors the name 04_train.py builds, so an already-finished cell is skipped
  # rather than retrained. Keep the two in step.
  local label="$init"
  [ "$parts" = "all" ] || label="${init}-${parts}"
  [ -z "$mt5" ] || label="${label}+${mt5}-mt5"
  local name="f${f}_local_${label}_${TUNE}_k${CTXK}_s${seed}"

  local args=(--fold "$f" --system local --init "$init" --init-parts "$parts"
              --tune "$TUNE" --ctx-k "$CTXK" --epochs "$EPOCHS" --seed "$seed")
  [ -z "$mt5" ] || args+=(--init-mt5 "$mt5")

  if [ -f "runs/$name/best.pt" ]; then
    echo "--- $name already trained"
  else
    if ! have_time_for_a_cell; then DEFERRED+=("$name"); return; fi
    wait_for_gpu || { DEFERRED+=("$name"); return; }
    echo "--- $name  ($(date +%H:%M), $(( $(secs_to_plan) / 60 )) min to plan-by)"
    set +e
    timeout -k 60 "$(secs_to_hard)" python3 scripts/04_train.py "${args[@]}"
    local rc=$?
    set -e
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
      echo "### $name hit the $HARD_STOP hard stop and was killed"
      FAILED+=("hard-stop $name"); return
    elif [ "$rc" -ne 0 ]; then
      FAILED+=("train $name"); return
    fi
  fi
  if [ ! -f "runs/$name/eval.json" ]; then
    # Scoring needs the card too (~4 GB, far less than training's 14.4). Without
    # its own window an eval walks straight into a full card and OOMs, which is
    # how a finished 23-minute training cell ends up with a checkpoint and no
    # result. GPU_EVAL_MIB is separate from GPU_NEED_MIB so a run that cannot be
    # trained can still be scored.
    wait_for_gpu "${GPU_EVAL_MIB:-6000}" || { DEFERRED+=("eval $name"); return; }
    timeout -k 60 "$(secs_to_hard)" python3 scripts/05_eval.py \
      --ckpt "runs/$name/best.pt" --conditions "$EVAL_CONDS" \
      || { FAILED+=("eval $name"); return; }
  fi
  DONE=$((DONE + 1))
}

hold_until "$START_AFTER"
set_deadlines
wait_for_our_own_queue || true
echo "### mechanism sweep: ${#CELLS[@]} cells, priority order  ($(date +%H:%M))"
for spec in "${CELLS[@]}"; do
  IFS='|' read -r f init parts mt5 seed <<< "$spec"
  run_cell "$f" "$init" "$parts" "$mt5" "$seed"
done

echo "### $DONE cell(s) completed  ($(date +%H:%M))"
if [ ${#DEFERRED[@]} -ne 0 ]; then
  echo "### ${#DEFERRED[@]} cell(s) not started — out of time before $PLAN_BY, not failures."
  echo "### Re-run this script after 09:00 tomorrow; finished cells are skipped."
  printf '  %s\n' "${DEFERRED[@]}"
fi
if [ ${#FAILED[@]} -ne 0 ]; then
  echo "### ${#FAILED[@]} cell(s) failed — rerun this script to retry just these:"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
[ ${#DEFERRED[@]} -ne 0 ] && { echo "### mechanism sweep PARTIAL"; exit 2; }
echo "### mechanism sweep complete  ($(date +%H:%M))"
