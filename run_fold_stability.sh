#!/usr/bin/env bash
# Is the fold-0 excursion a property of fold 0, or does the same configuration
# make the same excursion on any fold given a different draw?
#
# ---------------------------------------------------------------- the question
#
# On fold 0 the How2Sign and WLASL *pose branches* on a vanilla mT5-base reach
# 73.8% on the pronoun diagnostic, against ~56% for every other encoder-only row
# and 56.5% for `random`. Measured 8/15, the excursion is:
#
#   * seed-robust      73.8 / 73.5 / 72.5 (How2Sign, s0-s2 on fold 0)
#                      73.8 / 69.0 / 72.5 (WLASL,    s0-s2 on fold 0)
#   * fold-local       57.3 / 54.1 (How2Sign, folds 1 / 2, seed 0)
#   * visible in dev CE, which is measured on paragraphs disjoint from test:
#                      3.74-3.76 on fold 0 against 3.87-3.98 everywhere else
#   * a real use of the video: blanking the clip costs those cells +16 to +25
#                      points, against ~0 for every other encoder-only row
#   * absent on the lexical set: 43.6% against `random`'s 40.6% on fold 0
#
# So the model found a first/second-person pointing cue -- 我 92%, 你 88%, but
# 他 20% and 她 0% -- and found it on fold 0 only. Folds 1 and 2 have ONE seed
# each, so "it is the fold" currently rests on n=1 per fold. That is the gap
# these cells close.
#
#   block A: seeds 1 and 2 for both configurations on folds 1 and 2.
#            If all eight land at ~55%, fold 0 is genuinely the outlier and the
#            paper can say so. If any lands at ~73%, the excursion is bimodal
#            optimisation that fold 0 merely sampled three times out of three,
#            and the fold framing in the draft is wrong.
#
#   block B: same-fold controls. csl_daily-pose and random at seed 1 on fold 1,
#            so block A's spread has a same-fold yardstick that is not itself
#            one of the two excursion rows.
#
#   block C: diagnostic, NOT for the table -- see the --tag below. On fold 2 the
#            How2Sign-pose dev curve starts at 10.99 and needs six epochs to come
#            within half a nat of its own final value; fold 0 starts at 6.73 and
#            gets there in three. (An earlier version of this note said fold 2 was
#            "still descending at epoch 12". It is not -- every run here is flat
#            to within 0.006 nats over its last three epochs -- and the catching-up
#            reading is the one the curves support.) So the difference may be
#            optimisation budget rather than fold, for fold 2 at least: it cannot
#            be the story on fold 1, which settles as fast as fold 0 and still
#            does not depart. These
#            two cells re-run folds 1 and 2 at 20 epochs. They are off-protocol
#            and go to their own run directories; they answer "could folds 1-2
#            get there with more budget", which is a different question from
#            block A's, and neither may be pooled with the 12-epoch rows.
#
# ---------------------------------------------------------------- GPU sharing
#
# The colleague holds the card 21:00-09:00 by agreement. (PACLIC also had first
# claim on the morning; it finished 8/15, which is why EARLIEST moved from 11:00
# back to the agreement boundary.) So the gate is a condition, not a
# clock: after EARLIEST we sit out SETTLE_MIN without looking, then require the
# card idle for QUIET_MIN consecutive minutes. `touch .gpu-hold` keeps us off
# indefinitely. PLAN_BY is the scheduling limit -- no cell starts that we do not
# expect to finish by then -- and HARD_STOP is the safety net for one that
# overruns, enforced by `timeout`, not by an estimate.
#
#   nohup ./run_fold_stability.sh > runs/logs/fold_stability.log 2>&1 &
#   DRY=1 ./run_fold_stability.sh          # print the plan and exit
#   EARLIEST=09:00 ./run_fold_stability.sh # if PACLIC does not want the morning
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
TUNE=${TUNE:-full}
CTXK=${CTXK:-4}
EPOCHS=${EPOCHS:-12}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# 09:00 is the moment the colleague's 21:00-09:00 window ends, so it is the
# earliest this script may look at the card at all. It was 11:00 on the 8/15
# launch only because PACLIC had first claim on that morning; PACLIC is finished
# (8/15), so the window opens at the agreement boundary again. SETTLE/QUIET stay:
# they are now the handover buffer for an overnight job that runs past 09:00.
# QUIET_MIN was 15 until 8/17; shortened to 5 on request, the reservation being
# wider than it needed to be. It is the one guard that distinguishes "their job
# ended" from "their job is between phases", so if we ever take the card out
# from under a run that was only pausing, this is the number to put back.
EARLIEST=${EARLIEST:-09:00}
SETTLE_MIN=${SETTLE_MIN:-20}
QUIET_MIN=${QUIET_MIN:-5}
HOLD_FILE=${HOLD_FILE:-.gpu-hold}
PLAN_BY=${PLAN_BY:-20:00}
HARD_STOP=${HARD_STOP:-20:45}
CELL_BUDGET_MIN=${CELL_BUDGET_MIN:-30}
GPU_NEED_MIB=${GPU_NEED_MIB:-16000}
GPU_EVAL_MIB=${GPU_EVAL_MIB:-6000}
DRY=${DRY:-}

#   fold | init | parts | init-mt5 | seed | epochs | tag
#
# Field 4 is the second checkpoint: with `--init X --init-parts pose --init-mt5 Y`
# the pose branch comes from X and the mT5 decoder from Y, which is what the
# factorial's cross cells are. Empty means a single checkpoint, as before.
# (Same field order as run_mechanism_sweep2.sh, deliberately.)
CELLS=(
  # ---- block 0: the factorial off fold 0.  ADDED 8/15 late, and put first.
  # Every cell the body's encoder/decoder decomposition rests on is currently
  # fold 0, seed 0, n=330 on the content-word set — including the comparison
  # that matters most now: with the decoder held at CSL-Daily's, a CSL pose
  # branch scores 56.4% against an ASL one's 45.2/46.7 and reads the clip
  # (+20.0 against +3.9/+6.1). That is the one result in the paper the pronoun
  # items cannot produce by construction, and it has never been replicated on
  # another fold. It outranks block A: block A explains an appendix excursion
  # in the diagnostic we may be demoting.
  "1|csl_stage1|pose|csl_daily|0|12|"
  "2|csl_stage1|pose|csl_daily|0|12|"
  "1|how2sign|pose|csl_daily|0|12|"
  "2|how2sign|pose|csl_daily|0|12|"
  "1|wlasl|pose|csl_daily|0|12|"
  "2|wlasl|pose|csl_daily|0|12|"
  "1|csl_daily|pose|csl_stage1|0|12|"
  "2|csl_daily|pose|csl_stage1|0|12|"
  "1|csl_daily|pose|how2sign|0|12|"
  "2|csl_daily|pose|how2sign|0|12|"
  # ---- block A: does the excursion ever happen off fold 0?
  "1|how2sign|pose||1|12|"
  "1|wlasl|pose||1|12|"
  "2|how2sign|pose||1|12|"
  "2|wlasl|pose||1|12|"
  "1|how2sign|pose||2|12|"
  "1|wlasl|pose||2|12|"
  "2|how2sign|pose||2|12|"
  "2|wlasl|pose||2|12|"
  # ---- block B: same-fold controls
  "1|csl_daily|pose||1|12|"
  "1|random|all||1|12|"
  # ---- block C: budget diagnostic, off-protocol, own directories
  "1|how2sign|pose||0|20|f1_local_how2sign-pose_full_k4_s0_e20"
  "2|how2sign|pose||0|20|f2_local_how2sign-pose_full_k4_s0_e20"
)

# ------------------------------------------------------------------ the clocks
#
# Both deadlines are computed AFTER the hold, never at launch. This script is
# meant to be started the evening before and parked overnight; measuring against
# a deadline from the previous calendar day puts every deadline in the past and
# refuses every cell, which is how a queue that looks scheduled quietly does
# nothing. run_mechanism_sweep2.sh carries the same note for the same reason.
PLAN_TS=""; HARD_TS=""
set_deadlines() {
  PLAN_TS=$(date -d "today $PLAN_BY" +%s 2>/dev/null) || {
    echo "### PLAN_BY='$PLAN_BY' unparseable — refusing to run without a deadline"; exit 1; }
  HARD_TS=$(date -d "today $HARD_STOP" +%s 2>/dev/null) || {
    echo "### HARD_STOP='$HARD_STOP' unparseable — refusing to run"; exit 1; }
  if [ "$PLAN_TS" -le "$(date +%s)" ]; then
    echo "### plan-by $PLAN_BY already past ($(date +%H:%M)) — nothing can run today"; exit 2
  fi
  echo "### plan-by $PLAN_BY (no cell starts with under ${CELL_BUDGET_MIN} min left); hard stop $HARD_STOP"
}
secs_to_plan() { echo $(( PLAN_TS - $(date +%s) )); }
secs_to_hard() { echo $(( HARD_TS - $(date +%s) )); }
have_time()    { [ "$(secs_to_plan)" -ge $(( CELL_BUDGET_MIN * 60 )) ]; }
have_time_eval() { [ "$(secs_to_plan)" -ge 300 ]; }

hold_until_earliest() {
  local target now remain
  target=$(date -d "today $EARLIEST" +%s 2>/dev/null) || {
    echo "### EARLIEST='$EARLIEST' unparseable — refusing to guess"; exit 1; }
  # Launched after EARLIEST has passed means the next window is tomorrow's, not
  # "now": at 19:45 today, `today 11:00` is in the past and a naive check would
  # take the card during the colleague's evening.
  [ "$target" -le "$(date +%s)" ] && target=$(date -d "tomorrow $EARLIEST" +%s)
  while :; do
    remain=$(( target - $(date +%s) ))
    [ "$remain" -le 0 ] && break
    echo "### holding until $EARLIEST before looking at the card: $(( remain / 60 )) min  ($(date +%H:%M))"
    if [ "$remain" -lt 1800 ]; then sleep "$remain"; else sleep 1800; fi
  done
  echo "### $EARLIEST reached ($(date +%H:%M))"
}

gpu_busy_procs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . ; }
gpu_free_mib()   { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

wait_for_quiet_card() {
  local quiet=0 procs free
  echo "### settling ${SETTLE_MIN} min in case an overnight job runs past $EARLIEST  ($(date +%H:%M))"
  sleep $(( SETTLE_MIN * 60 ))
  while :; do
    have_time || { echo "### plan-by reached while waiting for a clear card; nothing started"; return 1; }
    if [ -f "$HOLD_FILE" ]; then
      echo "### $HOLD_FILE present — holding off  ($(date +%H:%M))"; quiet=0; sleep 300; continue
    fi
    procs=$(gpu_busy_procs); free=$(gpu_free_mib)
    if [ "${procs:-1}" -eq 0 ] && [ "${free:-0}" -ge "$GPU_NEED_MIB" ]; then
      quiet=$(( quiet + 1 ))
      [ "$quiet" -ge "$QUIET_MIN" ] && { echo "### card idle ${QUIET_MIN} min straight — taking it  ($(date +%H:%M))"; return 0; }
    else
      [ "$quiet" -gt 0 ] && echo "### card busy again after ${quiet} min quiet (${procs} procs, ${free} MiB free) — restarting the count"
      quiet=0
    fi
    sleep 60
  done
}

# A cell that starts mid-sweep still has to find its own window: the colleague or
# PACLIC can come back between cells, and walking into a full card OOMs.
wait_for_gpu() {
  local need=${1:-$GPU_NEED_MIB} waited=0 free
  while :; do
    [ -f "$HOLD_FILE" ] || {
      free=$(gpu_free_mib)
      [ "${free:-0}" -ge "$need" ] && { [ "$waited" -gt 0 ] && echo "### card free (${free} MiB) after $(( waited / 60 )) min"; return 0; }
    }
    have_time || return 1
    [ $(( waited % 900 )) -eq 0 ] && echo "### waiting for ${need} MiB  ($(date +%H:%M))"
    sleep 60; waited=$(( waited + 60 ))
  done
}

DONE=0; FAILED=(); DEFERRED=()

run_eval() {  # run_eval <name> <outfile> <data> <conditions>
  local name=$1 out=$2 data=$3 conds=$4
  [ -f "runs/$name/$out" ] && return 0
  have_time_eval || { DEFERRED+=("eval $out $name"); return 1; }
  wait_for_gpu "$GPU_EVAL_MIB" || { DEFERRED+=("eval $out $name"); return 1; }
  if timeout -k 60 "$(secs_to_hard)" "$PY" scripts/05_eval.py \
       --ckpt "runs/$name/best.pt" --data "$data" \
       --conditions "$conds" --out "runs/$name/$out"; then
    return 0
  fi
  echo "### FAILED eval $out $name"
  rm -f "runs/$name/$out"   # never leave a partial file that looks complete
  FAILED+=("eval $out $name")
  return 1
}

run_cell() {
  local f=$1 init=$2 parts=$3 mt5=$4 seed=$5 epochs=$6 tag=$7
  local label="$init"
  [ "$parts" = "all" ] || label="${init}-${parts}"
  # Must match 04_train.py's own naming (`init = f"{init}+{a.init_mt5}-mt5"`),
  # or the "already trained" check below looks in the wrong directory and every
  # finished cell is silently retrained.
  [ -z "$mt5" ] || label="${label}+${mt5}-mt5"
  local name=${tag:-"f${f}_local_${label}_${TUNE}_k${CTXK}_s${seed}"}

  if [ -f "runs/$name/best.pt" ]; then
    echo "--- $name already trained"
  else
    have_time    || { DEFERRED+=("$name"); return; }
    wait_for_gpu || { DEFERRED+=("$name"); return; }
    echo "--- $name  ($(date +%H:%M), $(( $(secs_to_plan) / 60 )) min to plan-by)"
    local args=(--fold "$f" --system local --init "$init" --init-parts "$parts"
                --tune "$TUNE" --ctx-k "$CTXK" --epochs "$epochs" --seed "$seed")
    [ -z "$mt5" ] || args+=(--init-mt5 "$mt5")
    [ -z "$tag" ] || args+=(--tag "$tag")
    timeout -k 60 "$(secs_to_hard)" "$PY" scripts/04_train.py "${args[@]}"
    local rc=$?
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
      echo "### $name hit the $HARD_STOP hard stop and was killed"
      FAILED+=("hard-stop $name"); return
    elif [ "$rc" -ne 0 ]; then
      FAILED+=("train $name"); return
    fi
  fi
  # The three readouts the paper quotes, in the order it needs them. eval_blank
  # is the one-factor ablation the Delta column is read off; eval_lex is the
  # content-word set that says whether the excursion is pronoun-specific.
  run_eval "$name" eval.json       data     local,text_only  || return
  run_eval "$name" eval_blank.json data     blank_plain      || return
  run_eval "$name" eval_lex.json   data/lex local,blank_plain || return
  DONE=$(( DONE + 1 ))
}

echo "### fold-stability sweep: ${#CELLS[@]} cells, priority order"
echo "### earliest $EARLIEST, then settle ${SETTLE_MIN}m and require ${QUIET_MIN}m idle"
if [ -n "$DRY" ]; then
  for spec in "${CELLS[@]}"; do
    IFS='|' read -r f init parts mt5 seed epochs tag <<< "$spec"
    label="$init"; [ "$parts" = "all" ] || label="${init}-${parts}"
    [ -z "$mt5" ] || label="${label}+${mt5}-mt5"
    name=${tag:-"f${f}_local_${label}_${TUNE}_k${CTXK}_s${seed}"}
    printf '  %-46s %s\n' "$name" "$([ -f "runs/$name/best.pt" ] && echo 'trained' || echo "${epochs}ep, to train")"
  done
  exit 0
fi

hold_until_earliest
set_deadlines
wait_for_quiet_card || exit 0

for spec in "${CELLS[@]}"; do
  IFS='|' read -r f init parts mt5 seed epochs tag <<< "$spec"
  run_cell "$f" "$init" "$parts" "$mt5" "$seed" "$epochs" "$tag"
done

echo
echo "### $DONE cell(s) complete  ($(date +%H:%M))"
if [ ${#DEFERRED[@]} -ne 0 ]; then
  echo "### ${#DEFERRED[@]} not started — out of time before $PLAN_BY, not failures."
  echo "### Re-run this script tomorrow; finished cells are skipped."
  printf '  %s\n' "${DEFERRED[@]}"
fi
if [ ${#FAILED[@]} -ne 0 ]; then
  echo "### ${#FAILED[@]} failed — re-run to retry just these:"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
[ ${#DEFERRED[@]} -ne 0 ] && exit 2
echo "### fold-stability sweep complete  ($(date +%H:%M))"
