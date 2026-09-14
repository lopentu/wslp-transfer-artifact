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
  # ---- block 3: where the pairing lives
  "0|csl_daily|proj||0"
  "0|csl_daily|mt5_proj||0"
  "0|csl_daily|pose_noproj||0"
  # ---- block 4: is the fold-0 excursion seed luck?
  "0|how2sign|pose||1"
  "0|wlasl|pose||1"
  "0|how2sign|pose||2"
  "0|wlasl|pose||2"
  # ---- block 5: seed variance on the headline rows
  "0|csl_daily|all||1"
  "0|random|all||1"
  "0|how2sign|all||1"
  "0|csl_daily|all||2"
  "0|random|all||2"
  "0|how2sign|all||2"
  # ---- block 6: seed variance on the reported encoder-only row
  "0|csl_daily|pose||1"
  "0|csl_daily|pose||2"
)

PLAN_TS=""; HARD_TS=""
if [ -n "$PLAN_BY" ]; then
  # Deliberately "today": past PLAN_BY the deadline is in the past and every
  # cell is refused. Rolling to tomorrow would silently invert the agreement.
  PLAN_TS=$(date -d "today $PLAN_BY" +%s 2>/dev/null) || {
    echo "### PLAN_BY='$PLAN_BY' unparseable — refusing to run without a deadline"; exit 1; }
  HARD_TS=$(date -d "today $HARD_STOP" +%s 2>/dev/null) || {
    echo "### HARD_STOP='$HARD_STOP' unparseable"; exit 1; }
  echo "### plan-by $PLAN_BY (no cell starts with under ${CELL_BUDGET_MIN} min left); hard stop $HARD_STOP"
fi

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
  local waited=0 free
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge "$GPU_NEED_MIB" ]; then
      [ "$waited" -gt 0 ] && echo "### GPU free (${free} MiB) after $((waited / 60)) min"
      return 0
    fi
    have_time_for_a_cell || { echo "### plan-by reached while waiting for a GPU window"; return 1; }
    if [ "$waited" -ge "$GPU_WAIT_MAX" ]; then
      echo "### gave up: ${free} MiB free, need ${GPU_NEED_MIB}, waited $((waited / 60)) min"
      return 1
    fi
    if [ $((waited % 900)) -eq 0 ]; then
      echo "### waiting for the GPU: ${free} MiB free, need ${GPU_NEED_MIB} MiB  ($(date +%H:%M))"
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
    timeout -k 60 "$(secs_to_hard)" python3 scripts/05_eval.py \
      --ckpt "runs/$name/best.pt" --conditions "$EVAL_CONDS" \
      || { FAILED+=("eval $name"); return; }
  fi
  DONE=$((DONE + 1))
}

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
