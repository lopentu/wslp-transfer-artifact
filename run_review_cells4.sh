#!/usr/bin/env bash
# The cells the three 8/22 reviews ask for, in the order their value per GPU-hour
# puts them. Deadline is 8/24, so the queue has to be able to stop cleanly at any
# point and still leave the paper better than it found it.
#
# ------------------------------------------------------------------- block P
# Forward passes only, minutes each, and all three reviews name the first one as
# the single highest-return experiment outstanding. §4 says two readings of the
# lm_head result remain open -- text-side initialization changes what the network
# LEARNS from the video, or it leaves a readout that cannot EXPRESS what the
# visual branch already encodes -- and then names the control that separates them
# without doing it: swap `lm_head` AFTER fine-tuning instead of before.
#
#   how2sign  <- csl_daily's fine-tuned head    does a good readout rescue it?
#   csl_daily <- how2sign's  fine-tuned head    does a bad one break it?
#   how2sign  <- the head of the INIT-swapped   is it CSL-Daily's head, or any
#                cell that was already rescued  head trained in a good trajectory?
#   how2sign  <- its own head from seed 1       the null: what does replacing the
#                                               head with a compatible one cost?
#
# The last is what makes the other three readable. A 192.1M-parameter tensor is a
# fifth of the mT5 half, and two independently fine-tuned models do not share a
# coordinate system; without the seed-1 control, any drop could be that rather
# than anything about written language.
#
# Block P also adds `swap_within`, the wrong-clip control drawn from the target's
# OWN paragraph. `swap_plain` takes its donor from a different paragraph, so it
# changes signer, session and topic along with the clip-sentence correspondence,
# and a reviewer is right that part of the 22.1 points could be the former. The
# within-paragraph version holds signer and session fixed. It matches duration
# less tightly (280 ms against 10 ms; a paragraph has only ~12 utterances to
# choose from), so the two bracket the effect rather than one replacing the other.
#
# ------------------------------------------------------------------- block T
# Training, ~30 min a cell. Two reviews independently ask for the same missing
# level of the head factor, and it is the one that decides the paper's framing.
# Table 1c has How2Sign+CSL head = 48.5 and CSL+How2Sign head = 24.3, so a
# Chinese head is better than an English one -- but there is no NEUTRAL head, and
# without it "a Chinese output projection lets visual transfer be used" and
# "removing an English-specialized one is enough" fit the data equally well. The
# paper's own RAND-VIS row (40.1), whose head is mT5-base's, suggests the second.
#
#   how2sign  + mt5_base lm_head    the neutral third level, recipient ASL
#   csl_daily + mt5_base lm_head    the neutral third level, recipient CSL
#
# Then the control §4 calls unavailable and a review points out we can build:
#
#   how2sign  + tsl_text lm_head    a Chinese-adapted head that never saw a sign
#                                   language (scripts/23_text_lm.py). This is the
#                                   only cell in the paper where written-language
#                                   match is SET rather than inherited.
#
# Then the two seeds that finish the headline 2x2. Three of its four cells have
# seeds 0/1/2 on fold 0; `csl_daily-mt5` has seed 0 alone, so the 15.5-point
# interaction cannot be quoted at any seed but 0. Both reviews ask for this
# before any new checkpoint.
#
#   csl_daily-mt5 s1, s2
#
# Then sufficiency, the mirror of the existing necessity result: `lm_head` alone
# on an mT5-base body, with CSL-Daily's visual half. Table 1c shows the CSL head
# is NECESSARY (removing it costs 23 points); this asks whether it is SUFFICIENT.
#
#   csl_daily-pose + csl_daily lm_head
#
# Last, and only if the queue gets there: label smoothing. A review's sharpest
# alternative reading of the whole paper is that this is optimization collapse
# rather than conditional transfer, and asks where the hyperparameters were
# chosen and whether a lower label smoothing lets How2Sign escape. 0.2 is
# upstream's value, so it was chosen on none of our cells -- but that is an
# argument about provenance and this is a measurement.
#
#   how2sign, label smoothing 0.0
#
# ---------------------------------------------------------------- GPU sharing
# The colleague holds the card 21:00-09:00 and the 8/19-8/21 release has expired,
# so the boundary is back. PLAN_BY is the last moment a cell may START, HARD_STOP
# is a `timeout` rather than an estimate, and a cell that cannot finish before
# PLAN_BY is deferred to tomorrow rather than begun.
#
#   cp run_review_cells4.sh .run_rc4.running.sh   # bash reads a script by byte
#   nohup ./.run_rc4.running.sh > runs/logs/rc4.log 2>&1 &   # offset; never edit
#   DRY=1 ./run_review_cells4.sh                             # a running one
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
TUNE=${TUNE:-full}
CTXK=${CTXK:-4}
EPOCHS=${EPOCHS:-12}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

EARLIEST=${EARLIEST:-}
HOLD_FILE=${HOLD_FILE:-.gpu-hold}
# 20:00 is the agreement, and a cell needs 30 min, so nothing starts after 19:30.
# HARD_STOP at 20:00 exactly: the `timeout` must not run into the colleague's
# window even if a cell misbehaves.
PLAN_BY=${PLAN_BY:-2026-08-22 19:30}
HARD_STOP=${HARD_STOP:-2026-08-22 20:00}
CELL_BUDGET_MIN=${CELL_BUDGET_MIN:-32}
GPU_NEED_MIB=${GPU_NEED_MIB:-16000}
GPU_EVAL_MIB=${GPU_EVAL_MIB:-7000}
DRY=${DRY:-}

CLIP_CONDS=${CLIP_CONDS:-local,blank_plain,swap_plain@0,swap_plain@1,swap_plain@2,shuffle_frames}
# The post-hoc graft and the within-paragraph swap are read off the same four
# conditions: accuracy and the score margin, `local` against three wrong-clip
# draws. `blank_plain` is deliberately absent -- it is out of distribution, which
# is the finding that produced `swap_plain` in the first place.
POST_CONDS=${POST_CONDS:-local,swap_plain@0,swap_plain@1,swap_plain@2}
WITHIN_CONDS=${WITHIN_CONDS:-local,swap_within@0,swap_within@1,swap_within@2}

# ---------------------------------------------------------------- block P specs
#   recipient run | donor run (best.pt) | outfile
POSTHOC=(
  "f0_local_how2sign_full_k4_s0|f0_local_csl_daily_full_k4_s0|eval_posthoc_csldaily.json"
  "f0_local_csl_daily_full_k4_s0|f0_local_how2sign_full_k4_s0|eval_posthoc_how2sign.json"
  "f0_local_how2sign_full_k4_s0|f0_local_how2sign+csl_daily-lm_head_full_k4_s0|eval_posthoc_rescued.json"
  "f0_local_how2sign_full_k4_s0|f0_local_how2sign_full_k4_s1|eval_posthoc_selfseed.json"
  # Folds 1 and 2 of the two primary directions, so the post-hoc row can be
  # pooled over the same 940 items as every other row in Table 1 instead of
  # being the paper's one 330-item claim.
  "f1_local_how2sign_full_k4_s0|f1_local_csl_daily_full_k4_s0|eval_posthoc_csldaily.json"
  "f2_local_how2sign_full_k4_s0|f2_local_csl_daily_full_k4_s0|eval_posthoc_csldaily.json"
  "f1_local_csl_daily_full_k4_s0|f1_local_how2sign_full_k4_s0|eval_posthoc_how2sign.json"
  "f2_local_csl_daily_full_k4_s0|f2_local_how2sign_full_k4_s0|eval_posthoc_how2sign.json"
)

# Runs that get the within-paragraph wrong clip. The two rows §3.2 quotes, their
# untrained-visual control, and the single-tensor cell whose clip sensitivity
# §3.4's last paragraph rests on.
WITHIN=(
  f0_local_csl_daily_full_k4_s0
  f0_local_how2sign_full_k4_s0
  f0_local_random_full_k4_s0
  f0_local_how2sign+csl_daily-lm_head_full_k4_s0
  f1_local_csl_daily_full_k4_s0
  f1_local_how2sign_full_k4_s0
  f2_local_csl_daily_full_k4_s0
  f2_local_how2sign_full_k4_s0
)

# ---------------------------------------------------------------- block T specs
#   fold | init | parts | init-mt5 | mt5-parts | seed | epochs | tag | extra args
CELLS=(
  "0|how2sign|all|mt5_base|lm_head|0|12||"
  "0|csl_daily|all|mt5_base|lm_head|0|12||"
  "0|how2sign|all|tsl_text|lm_head|0|12||"
  "0|csl_daily-mt5|-|||1|12||"
  "0|csl_daily-mt5|-|||2|12||"
  "0|csl_daily|pose|csl_daily|lm_head|0|12||"
  "0|how2sign|all|||0|12|f0_local_how2sign_ls00_full_k4_s0|--label-smoothing 0.0"
)

PLAN_TS=""; HARD_TS=""
set_deadlines() {
  PLAN_TS=$(date -d "$PLAN_BY" +%s 2>/dev/null) || {
    echo "### PLAN_BY='$PLAN_BY' unparseable — refusing to run without a deadline"; exit 1; }
  HARD_TS=$(date -d "$HARD_STOP" +%s 2>/dev/null) || {
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
  [ -z "$EARLIEST" ] && return 0
  target=$(date -d "$EARLIEST" +%s 2>/dev/null) || {
    echo "### EARLIEST='$EARLIEST' unparseable — refusing to guess"; exit 1; }
  [ "$target" -le "$(date +%s)" ] && return 0
  while :; do
    remain=$(( target - $(date +%s) ))
    [ "$remain" -le 0 ] && break
    echo "### holding until $EARLIEST: $(( remain / 60 )) min  ($(date +%H:%M))"
    if [ "$remain" -lt 1800 ]; then sleep "$remain"; else sleep 1800; fi
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
    [ $(( waited % 900 )) -eq 0 ] && echo "### waiting for ${need} MiB  ($(date +%H:%M))"
    sleep 60; waited=$(( waited + 60 ))
  done
}

DONE=0; FAILED=(); DEFERRED=()

cell_done() {
  local name=$1 want=$2 got
  [ -f "runs/$name/best.pt" ] && [ -f "runs/$name/log.json" ] || return 1
  got=$("$PY" -c "import json;print(json.load(open('runs/$name/log.json'))['epochs_done'])" 2>/dev/null) || return 1
  [ "${got:-0}" -ge "$want" ]
}

# run_eval <name> <outfile> <data> <conditions> [extra args...]
run_eval() {
  local name=$1 out=$2 data=$3 conds=$4; shift 4
  [ -f "runs/$name/$out" ] && return 0
  have_time_eval || { DEFERRED+=("eval $out $name"); return 1; }
  wait_for_gpu "$GPU_EVAL_MIB" || { DEFERRED+=("eval $out $name"); return 1; }
  timeout -k 60 "$(secs_to_hard)" "$PY" scripts/05_eval.py \
       --ckpt "runs/$name/best.pt" --data "$data" --batch "${EVAL_BATCH:-4}" \
       --workers "${EVAL_WORKERS:-2}" --conditions "$conds" \
       --out "runs/$name/$out" "$@" 2>&1 | grep -v "^Loading weights"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ] && [ -f "runs/$name/$out" ]; then return 0; fi
  echo "### FAILED eval $out $name (exit $rc)"
  rm -f "runs/$name/$out"
  FAILED+=("eval $out $name")
  return 1
}

run_cell() {
  local f=$1 init=$2 parts=$3 mt5=$4 mt5parts=$5 seed=$6 epochs=$7 tag=$8 extra=$9
  # `-` means "the init name already carries its part suffix", which is how the
  # existing run directories are named for `csl_daily-mt5`: 04_train.py takes
  # --init csl_daily --init-parts mt5 and writes `csl_daily-mt5`. Passing the
  # suffixed name through here would produce `csl_daily-mt5-mt5`.
  local iparts="$parts"
  if [ "$parts" = "-" ]; then
    iparts="${init##*-}"; init="${init%-*}"
  fi
  local label="$init"
  [ "$iparts" = "all" ] || label="${init}-${iparts}"
  [ -z "$mt5" ] || label="${label}+${mt5}-${mt5parts}"
  local name=${tag:-"f${f}_local_${label}_${TUNE}_k${CTXK}_s${seed}"}

  if cell_done "$name" "$epochs"; then
    echo "--- $name already trained"
  else
    if [ -f "runs/$name/best.pt" ]; then
      echo "### $name has a best.pt but its log stops short of $epochs epochs — retraining"
    fi
    have_time    || { DEFERRED+=("$name"); return; }
    wait_for_gpu || { DEFERRED+=("$name"); return; }
    echo "--- $name  ($(date +%H:%M), $(( $(secs_to_plan) / 60 )) min to plan-by)"
    local args=(--fold "$f" --system local --init "$init" --init-parts "$iparts"
                --tune "$TUNE" --ctx-k "$CTXK" --epochs "$epochs" --seed "$seed")
    [ -z "$mt5" ] || args+=(--init-mt5 "$mt5" --init-mt5-parts "$mt5parts")
    [ -z "$tag" ] || args+=(--tag "$tag")
    # Unquoted on purpose: `extra` is a flag list from the table above, not user
    # input, and the flags are separate argv entries.
    # shellcheck disable=SC2086
    [ -z "$extra" ] || args+=($extra)
    timeout -k 60 "$(secs_to_hard)" "$PY" scripts/04_train.py "${args[@]}"
    local rc=$?
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
      echo "### $name hit the $HARD_STOP hard stop and was killed"
      FAILED+=("hard-stop $name"); return
    elif [ "$rc" -ne 0 ]; then
      FAILED+=("train $name"); return
    fi
  fi
  run_eval "$name" eval.json       data     local,text_only   || return
  run_eval "$name" eval_blank.json data     blank_plain       || return
  run_eval "$name" eval_lex.json   data/lex local,blank_plain || return
  run_eval "$name" eval_clip.json  data/lex "$CLIP_CONDS"     || return
  DONE=$(( DONE + 1 ))
}

if [ -n "$DRY" ]; then
  echo "### block P: post-hoc lm_head graft (forward passes only)"
  for spec in "${POSTHOC[@]}"; do
    IFS='|' read -r rcp donor out <<< "$spec"
    st=missing; [ -f "runs/$rcp/best.pt" ] && [ -f "runs/$donor/best.pt" ] && st="ready"
    [ -f "runs/$rcp/$out" ] && st="done"
    printf '  %-46s <- %-46s %s\n' "$rcp" "$donor" "$st"
  done
  echo "### block P: within-paragraph wrong clip"
  for n in "${WITHIN[@]}"; do
    st=missing; [ -f "runs/$n/best.pt" ] && st="ready"
    [ -f "runs/$n/eval_within.json" ] && st="done"
    printf '  %-52s %s\n' "$n" "$st"
  done
  echo "### block T: training"
  for spec in "${CELLS[@]}"; do
    IFS='|' read -r f init parts mt5 mt5parts seed epochs tag extra <<< "$spec"
    iparts="$parts"
    if [ "$parts" = "-" ]; then iparts="${init##*-}"; init="${init%-*}"; fi
    label="$init"; [ "$iparts" = "all" ] || label="${init}-${iparts}"
    [ -z "$mt5" ] || label="${label}+${mt5}-${mt5parts}"
    name=${tag:-"f${f}_local_${label}_${TUNE}_k${CTXK}_s${seed}"}
    printf '  %-52s %s %s\n' "$name" "$(cell_done "$name" "$epochs" && echo 'trained' || echo "${epochs}ep, to train")" "$extra"
  done
  exit 0
fi

hold_until_earliest
set_deadlines

echo "### block P1: post-hoc lm_head graft, ${#POSTHOC[@]} evals"
for spec in "${POSTHOC[@]}"; do
  IFS='|' read -r rcp donor out <<< "$spec"
  if [ ! -f "runs/$donor/best.pt" ]; then
    echo "### skip $out on $rcp — donor $donor has no best.pt"
    FAILED+=("donor-missing $donor"); continue
  fi
  echo "--- posthoc $rcp <- $donor  ($(date +%H:%M))"
  run_eval "$rcp" "$out" data/lex "$POST_CONDS" \
           --posthoc-head "runs/$donor/best.pt" && DONE=$(( DONE + 1 ))
done

echo
echo "### block P2: within-paragraph wrong clip, ${#WITHIN[@]} evals"
for n in "${WITHIN[@]}"; do
  echo "--- within $n  ($(date +%H:%M))"
  run_eval "$n" eval_within.json data/lex "$WITHIN_CONDS" && DONE=$(( DONE + 1 ))
done

echo
echo "### block T: ${#CELLS[@]} training cells"
for spec in "${CELLS[@]}"; do
  IFS='|' read -r f init parts mt5 mt5parts seed epochs tag extra <<< "$spec"
  run_cell "$f" "$init" "$parts" "$mt5" "$mt5parts" "$seed" "$epochs" "$tag" "$extra"
done

echo
echo "### $DONE unit(s) complete  ($(date +%H:%M))"
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
echo "### review-response sweep v4 complete  ($(date +%H:%M))"
exit 0
