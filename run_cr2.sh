#!/usr/bin/env bash
# Camera-ready round 2: the fold-0-only random-projection controls, extended to
# the three-fold protocol.
#
# ------------------------------------------------------------------ why these
# `rand_head_nm` was born in run_rc5.sh block 2 (8/22) as a fold-0 / seed-0
# control, and nothing since has revisited it: run_fold_stability.sh (8/17)
# predates the donor by five days and only ever covered the pose cells, and
# run_cr1.sh (9/13) filled folds 1/2 for exactly the rows reviewer 2 named --
# the permutation donor, OpenASL, WLASL. The random controls were never in any
# CELLS list at a fold other than 0. That is a scope decision that was recorded
# rather than forgotten (make_numbers.py's THREE_FOLD_ROWS omits HeadHowRandNm,
# and Table 2's caption daggers the row), so the paper is not currently wrong.
#
# It is, however, lopsided where it matters most. `19_bootstrap.py` pairs on
# shared items, so every contrast touching `rand_head_nm` collapses to fold 0:
#
#   perm vs mT5-base       n=940, 341 clusters   [-25.5, -17.5]
#   perm vs rand_head_nm   n=330, 115 clusters   [-4.8, +3.5]   <- fold 0 only
#   rand_head_nm vs base   n=330, 115 clusters   [-27.7, -12.8]
#
# The middle row is the p=0.892 null that section 3.3 is built around, and
# make_numbers.py's own comment insists it be quoted as an interval because the
# null IS the result. A null-as-result on one fold and 115 clusters is the
# weakest load-bearing number in the camera-ready. Three folds should take the
# half-width from ~4.1 points to ~2.4.
#
# The donors are fixed files (data/external/head_donors/rand_head_nm.pth,
# written 8/22), so folds 1 and 2 load the SAME random matrix that fold 0 did.
# The instrument is held constant and only the split moves -- identical in kind
# to how cr1 extended `mt5_base_perm`. If the null survives, section 3.3 gets a
# three-fold interval; if it does not, we find out rather than a reader.
#
# ------------------------------------------------------------------- ordering
# Priority is by what section 3.3 leans on. The two How2Sign `rand_head_nm`
# folds come first because they alone fix the headline null; the CSL-Daily twin
# (finding 0f's equivalence) second; the library-init scale control last, since
# it only supports a subsidiary "scale does not explain the gap" sentence.
#
# ---------------------------------------------------------------- GPU sharing
# The card was verified free at 03:40 on 9/16 (24112 MiB, no compute apps).
# `.gpu-hold` has been on disk since 9/13 -- cr1's launcher created it to stand
# cron_resume_v3.sh down and the queue drained without removing it. It is left
# exactly where it is, because it is still doing that job, and this queue gates
# on its own HOLD_FILE like cr1 did.
#
# ------------------------------------------------------------------- deadline
# Submission is 9/15 AoE = 9/16 20:00 Taipei. FINAL_STOP is 14:00, not 20:00:
# the last six hours belong to regenerating numbers.tex, rebuilding, and
# pagecheck.sh, and a cell that lands at 19:50 cannot be written up. Measured
# cadence in cr1 was 23 min per cell including all five evals, so six cells is
# about 2h20m against a ten-hour window.
#
#   cp run_cr2.sh .run_cr2.running.sh && chmod +x .run_cr2.running.sh
#   nohup ./.run_cr2.running.sh >> runs/logs/cr2.log 2>&1 &
#   DRY=1 ./run_cr2.sh
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
TUNE=${TUNE:-full}
CTXK=${CTXK:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

DAY_START=${DAY_START:-00:00}     # card is ours from here
PLAN_TIME=${PLAN_TIME:-13:00}     # last moment a cell may start
STOP_TIME=${STOP_TIME:-14:00}     # hard `timeout` bounding each cell
FINAL_STOP=${FINAL_STOP:-2026-09-16 14:00}   # after this, write-up time
CELL_BUDGET_MIN=${CELL_BUDGET_MIN:-40}
GPU_NEED_MIB=${GPU_NEED_MIB:-16000}
GPU_EVAL_MIB=${GPU_EVAL_MIB:-7000}
HOLD_FILE=${HOLD_FILE:-.gpu-hold.bypassed-for-cr2}
DRY=${DRY:-}

CLIP_CONDS=${CLIP_CONDS:-local,blank_plain,swap_plain@0,swap_plain@1,swap_plain@2,shuffle_frames}

#   fold | init | parts | init-mt5 | mt5-parts | seed | epochs | tag
CELLS=(
  # -- the headline null: section 3.3's "a permuted donor is indistinguishable
  #    from a norm-matched random one" is currently 330 items / 115 clusters.
  "1|how2sign|all|rand_head_nm|lm_head|0|12|"
  "2|how2sign|all|rand_head_nm|lm_head|0|12|"
  # -- finding 0f's equivalence on the WORKING model: on CSL-Daily a random
  #    norm-matched head and How2Sign's fine-tuned one are the same thing.
  "1|csl_daily|all|rand_head_nm|lm_head|0|12|"
  "2|csl_daily|all|rand_head_nm|lm_head|0|12|"
  # -- lowest priority: library init at 2.2x the row norm, which supports only
  #    the subsidiary "overall scale does not explain the gap" sentence.
  "1|how2sign|all|rand_head|lm_head|0|12|"
  "2|how2sign|all|rand_head|lm_head|0|12|"
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
  echo "### cr2: ${#CELLS[@]} cells, priority order; window ${DAY_START}-${STOP_TIME} daily, final stop $FINAL_STOP"
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

echo "### cr2 starts $(date '+%F %H:%M'): ${#CELLS[@]} cells, window ${DAY_START}-${STOP_TIME}, final stop $FINAL_STOP"
for spec in "${CELLS[@]}"; do
  IFS='|' read -r f init parts mt5 mt5parts seed epochs tag <<< "$spec"
  run_cell "$f" "$init" "$parts" "$mt5" "$mt5parts" "$seed" "$epochs" "$tag"
done

echo
echo "### cr2 done $(date '+%F %H:%M'): $DONE cell(s) complete"
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
