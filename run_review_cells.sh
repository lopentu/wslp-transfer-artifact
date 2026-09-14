#!/usr/bin/env bash
# The cells the three reviews ask for. Priority order, and the order is the
# argument: each block answers an objection that, unanswered, costs the paper a
# claim.
#
# ------------------------------------------------------------------- block A
# "The decoder's written language sets the level" is identified by ONE English
# decoder. Table 2's only English-decoder cell is How2Sign's, so what the data
# strictly supports today is "the How2Sign-trained decoder transfers worse", not
# "an English-output decoder transfers worse". Two reviewers name this as the
# single most valuable cell to add, and it is the paper's title-level claim.
#
#   CSL-Daily pose + OpenASL decoder      a second, independently pretrained
#                                         English sentence-level decoder
#   CSL-Daily pose + WLASL decoder        a third, and a different pretraining
#                                         task (isolated-sign recognition)
#
# If both land near the How2Sign cell's 25%, the claim is about output language
# and not about one checkpoint. If they scatter, the claim has to be narrowed to
# the checkpoint -- which we would rather discover ourselves than be told.
#
# ------------------------------------------------------------------- block B
# The encoder axis has two ASL branches on a Chinese decoder (How2Sign, WLASL)
# and one CSL branch. OpenASL is the missing third: three ASL encoders landing
# together supports the source-language reading, three scattering supports a
# pretraining-scale/domain reading. Either way the encoder claim gets its own
# replication instead of resting on two cells.
#
# ------------------------------------------------------------------- block C
# Every load-bearing cell is seed 0, and Appendix B shows the seed deciding a
# 13-17 point difference elsewhere in this same model. The two cells that carry
# the encoder-source effect get seeds 1 and 2 so the effect can be quoted with a
# spread rather than as one draw.
#
# ------------------------------------------------------------------- block D
# "Worse than no sign-language pretraining" is measured at a fixed 12 epochs, so
# it could be slower adaptation rather than a worse endpoint. Against that: the
# intact How2Sign dev curve is already flat (5.62/5.61/5.62/5.61 over the last
# four epochs of fold 0, a 0.006-nat drop across the last three), so this is a
# confirmation rather than an open question -- but it is cheap, and both readings
# are publishable. `random` at the same 24 epochs is the control that keeps the
# comparison matched: the claim is a ranking, so both rows must move together.
#
# Off-protocol cells get their own directories (`_e24`) and may never be pooled
# with the 12-epoch rows, exactly as `run_fold_stability.sh`'s block C.
#
# ---------------------------------------------------------------- GPU sharing
# Identical to run_fold_stability.sh, and for the same agreement: the colleague
# holds the card 21:00-09:00. EARLIEST is the earliest we may look at it, PLAN_BY
# the last moment a cell may start, HARD_STOP a `timeout` rather than an estimate.
#
#   nohup ./run_review_cells.sh > runs/logs/review_cells.log 2>&1 &
#   DRY=1 ./run_review_cells.sh
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
TUNE=${TUNE:-full}
CTXK=${CTXK:-4}
EPOCHS=${EPOCHS:-12}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

EARLIEST=${EARLIEST:-09:00}
SETTLE_MIN=${SETTLE_MIN:-0}
QUIET_MIN=${QUIET_MIN:-1}
HOLD_FILE=${HOLD_FILE:-.gpu-hold}
PLAN_BY=${PLAN_BY:-20:00}
HARD_STOP=${HARD_STOP:-20:45}
CELL_BUDGET_MIN=${CELL_BUDGET_MIN:-30}
GPU_NEED_MIB=${GPU_NEED_MIB:-16000}
GPU_EVAL_MIB=${GPU_EVAL_MIB:-7000}
DRY=${DRY:-}

# The wrong-clip conditions ride along on every new cell, so a cell trained today
# arrives with the same readout as the re-scored ones (see run_wrongclip.sh).
CLIP_CONDS=${CLIP_CONDS:-local,blank_plain,swap_plain@0,swap_plain@1,swap_plain@2,shuffle_frames}

#   fold | init | parts | init-mt5 | seed | epochs | tag
CELLS=(
  # ---- block A: a second and third English decoder, the identification cell
  "0|csl_daily|pose|openasl|0|12|"
  "1|csl_daily|pose|openasl|0|12|"
  "2|csl_daily|pose|openasl|0|12|"
  # ---- block B: the third ASL encoder on a Chinese decoder
  "0|openasl|pose|csl_daily|0|12|"
  "1|openasl|pose|csl_daily|0|12|"
  "2|openasl|pose|csl_daily|0|12|"
  # ---- block C: seeds for the two cells carrying the encoder-source effect
  "0|how2sign|pose|csl_daily|1|12|"
  "0|csl_stage1|pose|csl_daily|1|12|"
  "0|how2sign|pose|csl_daily|2|12|"
  "0|csl_stage1|pose|csl_daily|2|12|"
  # ---- block A': the third English decoder
  "0|csl_daily|pose|wlasl|0|12|"
  "1|csl_daily|pose|wlasl|0|12|"
  "2|csl_daily|pose|wlasl|0|12|"
  # ---- block D: budget control, off-protocol, own directories
  "0|how2sign|all||0|24|f0_local_how2sign_full_k4_s0_e24"
  "0|random|all||0|24|f0_local_random_full_k4_s0_e24"
  "1|how2sign|all||0|24|f1_local_how2sign_full_k4_s0_e24"
  "0|csl_daily|all||0|24|f0_local_csl_daily_full_k4_s0_e24"
)

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
have_time()      { [ "$(secs_to_plan)" -ge $(( CELL_BUDGET_MIN * 60 )) ]; }
have_time_eval() { [ "$(secs_to_plan)" -ge 300 ]; }

hold_until_earliest() {
  local target remain
  target=$(date -d "today $EARLIEST" +%s 2>/dev/null) || {
    echo "### EARLIEST='$EARLIEST' unparseable — refusing to guess"; exit 1; }
  # Started after EARLIEST has passed means today's window is already open, and
  # the deadline logic below is what stops us running into the colleague's.
  [ "$target" -le "$(date +%s)" ] && return 0
  while :; do
    remain=$(( target - $(date +%s) ))
    [ "$remain" -le 0 ] && break
    echo "### holding until $EARLIEST: $(( remain / 60 )) min  ($(date +%H:%M))"
    if [ "$remain" -lt 1800 ]; then sleep "$remain"; else sleep 1800; fi
  done
}

gpu_busy_procs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . ; }
gpu_free_mib()   { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

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
       --conditions "$conds" --out "runs/$name/$out" 2>&1 | grep -v "^Loading weights"; then
    return 0
  fi
  echo "### FAILED eval $out $name"
  rm -f "runs/$name/$out"
  FAILED+=("eval $out $name")
  return 1
}

run_cell() {
  local f=$1 init=$2 parts=$3 mt5=$4 seed=$5 epochs=$6 tag=$7
  local label="$init"
  [ "$parts" = "all" ] || label="${init}-${parts}"
  # Must match 04_train.py's own naming, or the "already trained" check looks in
  # the wrong directory and every finished cell is silently retrained.
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
  # The four readouts the paper quotes. eval_clip is the new one and comes last
  # because it is the longest; the three before it are what the existing tables
  # are generated from, so a cell that only gets those is still a usable row.
  run_eval "$name" eval.json       data     local,text_only   || return
  run_eval "$name" eval_blank.json data     blank_plain       || return
  run_eval "$name" eval_lex.json   data/lex local,blank_plain || return
  run_eval "$name" eval_clip.json  data/lex "$CLIP_CONDS"     || return
  DONE=$(( DONE + 1 ))
}

echo "### review-response sweep: ${#CELLS[@]} cells, priority order"
if [ -n "$DRY" ]; then
  for spec in "${CELLS[@]}"; do
    IFS='|' read -r f init parts mt5 seed epochs tag <<< "$spec"
    label="$init"; [ "$parts" = "all" ] || label="${init}-${parts}"
    [ -z "$mt5" ] || label="${label}+${mt5}-mt5"
    name=${tag:-"f${f}_local_${label}_${TUNE}_k${CTXK}_s${seed}"}
    printf '  %-52s %s\n' "$name" "$([ -f "runs/$name/best.pt" ] && echo 'trained' || echo "${epochs}ep, to train")"
  done
  exit 0
fi

hold_until_earliest
set_deadlines

for spec in "${CELLS[@]}"; do
  IFS='|' read -r f init parts mt5 seed epochs tag <<< "$spec"
  run_cell "$f" "$init" "$parts" "$mt5" "$seed" "$epochs" "$tag"
done

echo
echo "### $DONE cell(s) complete  ($(date +%H:%M))"
if [ ${#DEFERRED[@]} -ne 0 ]; then
  echo "### ${#DEFERRED[@]} not started — out of time before $PLAN_BY, not failures."
  printf '  %s\n' "${DEFERRED[@]}"
fi
if [ ${#FAILED[@]} -ne 0 ]; then
  echo "### ${#FAILED[@]} failed — re-run to retry just these:"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
[ ${#DEFERRED[@]} -ne 0 ] && exit 2
echo "### review-response sweep complete  ($(date +%H:%M))"
exit 0
