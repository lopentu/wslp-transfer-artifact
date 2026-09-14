#!/usr/bin/env bash
# Camera-ready round 1. The cells the two WSLP reviews ask for, in the order
# their absence would cost the paper most.
#
# ------------------------------------------------------------------- block P
# Reviewer 2's sharpest ask, and the only genuinely new INSTRUMENT here: a
# row-permuted mT5-base output projection. Every donor in the submitted paper
# varies the matrix's contents -- a fresh draw, a rescaling, a different
# fine-tune -- so each one moves the row statistics and the token
# correspondence together and neither can be credited alone. A row permutation
# separates them exactly: same multiset of rows, therefore the same Frobenius
# norm, row-norm distribution, singular values and conditioning as mT5-base,
# with only the row-to-token correspondence destroyed (scripts/27_head_donors.py
# asserts the invariants in float64). Rescue here means the recipient needs
# generic pretrained structure; no rescue means it needs rows sitting against
# the tokens they were trained for. Either answer is publishable and the paper
# currently cannot state either.
#
# `mt5_base_permws` draws the permutation inside each CJK/Latin/other group, so
# per-script norm structure survives too. Lowest priority: it only sharpens an
# answer block P already gives, and the CJK/global rescaling donors already show
# per-script scale is not the mechanism.
#
# ------------------------------------------------------------------- block R
# Reviewer 2's main concern, which is scope rather than validity: the head
# result is a three-fold protocol on How2Sign and a fold-0 check everywhere
# else, so "a broader checkpoint-level phenomenon" rests on one deeply analysed
# recipient. OpenASL first (sentence-level ASL->English, the closest analogue to
# How2Sign, and its own-head rows are already three-fold), then WLASL, whose
# isolated-sign pretraining makes it the stronger generality test of the two.
# Both donors -- mT5-base and CSL-Daily -- because the fold-0 table quotes both.
#
# ---------------------------------------------------------------- GPU sharing
# Until 2026-09-13 a colleague held the card 21:00-09:00 by agreement and this
# queue ran 09:00-20:00, sleeping in between. Kevin released that booking for the
# camera-ready push, so the defaults below now span the whole day. The daily
# bounds are kept rather than deleted because the mechanism is still wanted: a
# sweep should refuse to START a cell it cannot finish, and every training call
# is wrapped in a `timeout` bounded by the time left, so a hung cell overruns
# nothing. If the card is ever shared again, pass the window instead of editing:
#   DAY_START=09:00 PLAN_TIME=19:20 STOP_TIME=20:00 ./run_cr1.sh
#
# FINAL_STOP is the deadline that still bites -- after it, the remaining time
# belongs to writing, not to more cells. Measured cadence is 22 min per cell;
# CELL_BUDGET_MIN stays at 40 because it must also cover a slow eval.
#
# Standing the daily cron down: cron_resume_v3.sh fires 09:05 and would launch a
# SECOND queue onto the same card if it ever caught this one between cells. It
# stands down on `.gpu-hold`, so this script's launcher creates that file and
# this script gates on a DIFFERENT one (HOLD_FILE below), exactly as rc5p-rc5s
# did. Remember to remove .gpu-hold when the queue drains.
#
#   cp run_cr1.sh .run_cr1.running.sh && chmod +x .run_cr1.running.sh
#   touch .gpu-hold
#   nohup ./.run_cr1.running.sh >> runs/logs/cr1.log 2>&1 &
#   DRY=1 ./run_cr1.sh
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
TUNE=${TUNE:-full}
CTXK=${CTXK:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

DAY_START=${DAY_START:-00:00}     # card is ours from here
PLAN_TIME=${PLAN_TIME:-23:30}     # last moment a cell may start
STOP_TIME=${STOP_TIME:-23:59}     # hard `timeout` bounding each cell
FINAL_STOP=${FINAL_STOP:-2026-09-14 20:00}   # after this, write-up time
CELL_BUDGET_MIN=${CELL_BUDGET_MIN:-40}
GPU_NEED_MIB=${GPU_NEED_MIB:-16000}
GPU_EVAL_MIB=${GPU_EVAL_MIB:-7000}
HOLD_FILE=${HOLD_FILE:-.gpu-hold.bypassed-for-cr1}
DRY=${DRY:-}

CLIP_CONDS=${CLIP_CONDS:-local,blank_plain,swap_plain@0,swap_plain@1,swap_plain@2,shuffle_frames}

#   fold | init | parts | init-mt5 | mt5-parts | seed | epochs | tag
CELLS=(
  # -- block P, fold 0 first: one run answers the mechanism question at all.
  "0|how2sign|all|mt5_base_perm|lm_head|0|12|"
  # -- block R, OpenASL. Own-head folds 1/2 already exist, so these four cells
  #    complete the three-fold protocol for a second recipient.
  "1|openasl|all|mt5_base|lm_head|0|12|"
  "2|openasl|all|mt5_base|lm_head|0|12|"
  "1|openasl|all|csl_daily|lm_head|0|12|"
  "2|openasl|all|csl_daily|lm_head|0|12|"
  # -- block P, remaining folds: makes the permutation a three-fold row too,
  #    so it can be quoted with an interval instead of as one draw.
  "1|how2sign|all|mt5_base_perm|lm_head|0|12|"
  "2|how2sign|all|mt5_base_perm|lm_head|0|12|"
  # -- block R, WLASL: a third recipient, and a different pretraining task.
  "1|wlasl|all|mt5_base|lm_head|0|12|"
  "2|wlasl|all|mt5_base|lm_head|0|12|"
  "1|wlasl|all|csl_daily|lm_head|0|12|"
  "2|wlasl|all|csl_daily|lm_head|0|12|"
  # -- block P tail, lowest priority: within-script permutation, fold 0.
  "0|how2sign|all|mt5_base_permws|lm_head|0|12|"
)

FINAL_TS=$(date -d "$FINAL_STOP" +%s 2>/dev/null) || {
  echo "### FINAL_STOP='$FINAL_STOP' unparseable — refusing to run without a deadline"; exit 1; }

# Today's window, recomputed on every call: this queue spans days, so a
# timestamp captured once at launch would enforce yesterday's deadline.
plan_ts() { date -d "$(date +%F) $PLAN_TIME" +%s; }
hard_ts() { date -d "$(date +%F) $STOP_TIME" +%s; }
open_ts() { date -d "$(date +%F) $DAY_START" +%s; }
secs_to_hard() { echo $(( $(hard_ts) - $(date +%s) )); }
have_time()      { [ "$(( $(plan_ts) - $(date +%s) ))" -ge $(( CELL_BUDGET_MIN * 60 )) ]; }
have_time_eval() { [ "$(( $(plan_ts) - $(date +%s) ))" -ge 300 ]; }

# Sleep until the card is ours again. Returns 1 when the next window would open
# after FINAL_STOP, which is how the queue ends rather than by running out of
# cells.
wait_for_window() {
  local now target
  while :; do
    now=$(date +%s)
    [ "$now" -ge "$FINAL_TS" ] && return 1
    if [ "$now" -lt "$(open_ts)" ]; then
      target=$(open_ts)                                   # before 09:00 today
    elif have_time; then
      return 0                                            # in the window
    else
      target=$(date -d "tomorrow $DAY_START" +%s)         # past 19:20
    fi
    [ "$target" -ge "$FINAL_TS" ] && return 1
    echo "### out of window, sleeping until $(date -d "@$target" '+%m-%d %H:%M')" \
         "($(( (target - now) / 60 )) min)"
    while [ "$(date +%s)" -lt "$target" ]; do sleep 300; done
  done
}

gpu_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

wait_for_gpu() {
  local need=${1:-$GPU_NEED_MIB} waited=0 free
  while :; do
    [ -f "$HOLD_FILE" ] || {
      free=$(gpu_free_mib)
      [ "${free:-0}" -ge "$need" ] && { [ "$waited" -gt 0 ] && echo "### card free (${free} MiB) after $(( waited / 60 )) min"; return 0; }
    }
    have_time || return 1
    [ $(( waited % 900 )) -eq 0 ] && echo "### waiting for ${need} MiB  ($(date '+%m-%d %H:%M'))"
    sleep 60; waited=$(( waited + 60 ))
  done
}

DONE=0; FAILED=(); DEFERRED=()

# `best.pt` is rewritten every epoch a run improves on dev, so its presence means
# "started", not "finished". `log.json`'s epochs_done is the only honest marker.
cell_done() {
  local name=$1 want=$2 got
  [ -f "runs/$name/best.pt" ] && [ -f "runs/$name/log.json" ] || return 1
  got=$("$PY" -c "import json;print(json.load(open('runs/$name/log.json'))['epochs_done'])" 2>/dev/null) || return 1
  [ "${got:-0}" -ge "$want" ]
}

run_eval() {  # run_eval <name> <outfile> <data> <conditions>
  local name=$1 out=$2 data=$3 conds=$4
  [ -f "runs/$name/$out" ] && return 0
  have_time_eval || { DEFERRED+=("eval $out $name"); return 1; }
  wait_for_gpu "$GPU_EVAL_MIB" || { DEFERRED+=("eval $out $name"); return 1; }
  timeout -k 60 "$(secs_to_hard)" "$PY" scripts/05_eval.py \
       --ckpt "runs/$name/best.pt" --data "$data" --batch "${EVAL_BATCH:-4}" \
       --workers "${EVAL_WORKERS:-2}" --conditions "$conds" \
       --out "runs/$name/$out" 2>&1 | grep -v "^Loading weights"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] && [ -f "runs/$name/$out" ] && return 0
  echo "### FAILED eval $out $name (exit $rc)"
  rm -f "runs/$name/$out"      # never leave a partial file that looks complete
  FAILED+=("eval $out $name")
  return 1
}

run_cell() {
  local f=$1 init=$2 parts=$3 mt5=$4 mt5parts=$5 seed=$6 epochs=$7 tag=$8
  local label="$init"
  [ "$parts" = "all" ] || label="${init}-${parts}"
  [ -z "$mt5" ] || label="${label}+${mt5}-${mt5parts}"
  local name=${tag:-"f${f}_local_${label}_${TUNE}_k${CTXK}_s${seed}"}

  if cell_done "$name" "$epochs"; then
    echo "--- $name already trained"
  else
    [ -f "runs/$name/best.pt" ] && \
      echo "### $name has a best.pt but its log stops short of $epochs epochs — retraining"
    wait_for_window || { DEFERRED+=("$name"); return; }
    have_time    || { DEFERRED+=("$name"); return; }
    wait_for_gpu || { DEFERRED+=("$name"); return; }
    echo "--- $name  ($(date '+%m-%d %H:%M'), $(( ($(plan_ts) - $(date +%s)) / 60 )) min to plan-by)"
    local args=(--fold "$f" --system local --init "$init" --init-parts "$parts"
                --tune "$TUNE" --ctx-k "$CTXK" --epochs "$epochs" --seed "$seed")
    [ -z "$mt5" ] || args+=(--init-mt5 "$mt5" --init-mt5-parts "$mt5parts")
    [ -z "$tag" ] || args+=(--tag "$tag")
    timeout -k 60 "$(secs_to_hard)" "$PY" scripts/04_train.py "${args[@]}"
    local rc=$?
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
      echo "### $name hit the $STOP_TIME hard stop and was killed"
      FAILED+=("hard-stop $name"); return
    elif [ "$rc" -ne 0 ]; then
      FAILED+=("train $name"); return
    fi
  fi
  # All five readouts, including the pointing-sign wrong-clip one. The submitted
  # paper's coverage is ragged precisely because that eval was run as a separate
  # sweep over whatever existed at the time (see run_rc5r.sh/run_rc5s.sh), and
  # make_numbers.py's coverage guard then refuses a macro for any cell missing
  # it. A cell that arrives complete cannot reopen that gap.
  run_eval "$name" eval.json           data     local,text_only   || return
  run_eval "$name" eval_blank.json     data     blank_plain       || return
  run_eval "$name" eval_lex.json       data/lex local,blank_plain || return
  run_eval "$name" eval_clip.json      data/lex "$CLIP_CONDS"     || return
  run_eval "$name" eval_clip_pron.json data     "$CLIP_CONDS"     || return
  DONE=$(( DONE + 1 ))
}

if [ -n "$DRY" ]; then
  echo "### cr1: ${#CELLS[@]} cells, priority order; window ${DAY_START}-${STOP_TIME} daily, final stop $FINAL_STOP"
  for spec in "${CELLS[@]}"; do
    IFS='|' read -r f init parts mt5 mt5parts seed epochs tag <<< "$spec"
    label="$init"; [ "$parts" = "all" ] || label="${init}-${parts}"
    [ -z "$mt5" ] || label="${label}+${mt5}-${mt5parts}"
    name=${tag:-"f${f}_local_${label}_${TUNE}_k${CTXK}_s${seed}"}
    miss=""
    for e in eval.json eval_blank.json eval_lex.json eval_clip.json eval_clip_pron.json; do
      [ -f "runs/$name/$e" ] || miss="$miss ${e%.json}"
    done
    printf '  %-56s %s\n' "$name" \
      "$(cell_done "$name" "$epochs" && echo "trained, missing:${miss:- none}" || echo "${epochs}ep, TO TRAIN")"
  done
  exit 0
fi

# Refuse to stack on a queue that is already holding the card.
#
# `pgrep -f` alone cannot do this. The `[0]4_train` bracket trick only stops
# pgrep matching ITSELF; any other process whose command line happens to carry
# the literal text matches too, and the launching shell is exactly such a
# process -- this guard's first version refused to start because the one-liner
# that launched it mentioned `04_train.py` further along the line. So match the
# pattern, then keep only PIDs that are really a python interpreter (`comm` is
# `bash` for a shell and `python3`/`pt_main_thread` for a torch job).
queue_running() {
  local pid comm
  for pid in $(pgrep -f "scripts/0[45]_(train|eval)\.py" 2>/dev/null); do
    [ "$pid" = "$$" ] || [ "$pid" = "$PPID" ] && continue
    comm=$(cat "/proc/$pid/comm" 2>/dev/null) || continue
    case "$comm" in python*|pt_main_thread) return 0 ;; esac
  done
  return 1
}
if queue_running; then
  echo "### a trainer or evaluator is already running — refusing to start a second queue"
  exit 1
fi

echo "### cr1 starts $(date '+%F %H:%M'): ${#CELLS[@]} cells, window ${DAY_START}-${STOP_TIME}, final stop $FINAL_STOP"
for spec in "${CELLS[@]}"; do
  IFS='|' read -r f init parts mt5 mt5parts seed epochs tag <<< "$spec"
  run_cell "$f" "$init" "$parts" "$mt5" "$mt5parts" "$seed" "$epochs" "$tag"
done

echo
echo "### cr1 done $(date '+%F %H:%M'): $DONE cell(s) complete"
if [ ${#DEFERRED[@]} -ne 0 ]; then
  echo "### ${#DEFERRED[@]} not started — out of window, not failures:"
  printf '  %s\n' "${DEFERRED[@]}"
fi
if [ ${#FAILED[@]} -ne 0 ]; then
  echo "### ${#FAILED[@]} failed — re-run to retry just these:"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
[ ${#DEFERRED[@]} -ne 0 ] && exit 2
exit 0
