#!/usr/bin/env bash
# The training half of the 8/22-review response (writings/fifth/suggestions).
#
# All three reviews converge on the same block, and Kevin flagged it explicitly:
# `how2sign + mt5_base lm_head` (47.9) and `csl_daily + mt5_base lm_head` (63.0)
# are the two cells everyone is asking about, and both are currently seed 0 with
# no mechanism control. Everything in blocks 1-3 is about those two cells; blocks
# 4-7 are the other training asks.
#
#   1  O-P1   the two neutral-head cells at seeds 1 and 2, fold 0. The headline
#             rescue is single-seed while the paper itself reports a 16.6-point
#             seed spread elsewhere, so the interval that matters ("retrain and
#             is it still there?") does not exist yet.
#   2  O-P3   a randomly-initialised lm_head. Distinguishes "removing the
#             maladapted matrix is what helps" from "mT5-base's multilingual
#             geometry is what helps" -- the single most informative control
#             either review names, because the three outcomes imply three
#             different papers. Run norm-matched AND library-init: T5's own init
#             lands at 2.2x mT5-base's row norm, so the unmatched version alone
#             would confound geometry with scale.
#   3  A-W2   the per-script scale controls at initialisation. how2sign's head
#             ends with CJK:Latin norm ratio 0.64 against mT5-base's 1.02, so
#             "the ASL head kills a Chinese contrastive set" may be nothing but
#             quiet CJK rows. `_cjk` restores the ratio, `_gl` the overall
#             radius, `_cjkgl` both. If any of them rescues, the paper's
#             directional-specialisation account is in trouble and we find out
#             rather than a reviewer.
#   4  O-P2   does the localisation replicate on the other two ASL checkpoints,
#             or is it a How2Sign pathology? OpenASL and WLASL + mt5_base head.
#   5  A-B1   the learning-rate sweep. Appendix G moved epochs and label
#             smoothing but never LR, and "12 epochs at 3e-5 cannot drag a
#             specialised 192M head back" is the sharpest alternative to the
#             adaptation account. Swept on BOTH arms: if 3e-4 lifts intact
#             How2Sign to the rescued cell's level the story changes, and if it
#             lifts both equally the gap is not an optimisation artefact.
#   6  A-B2   the gloss->Chinese donor cell. Needs data/external/gloss_lm.pth
#             from run_rc5f.sh, so it is gated on the file and skipped if absent.
#   7  A-B6   model-selection sensitivity: three cells retrained with
#             --save-last, so the same trajectory can be read at its last epoch
#             instead of its best-dev-CE one. Also a determinism check, since
#             best.pt should reproduce bit-for-bit at a fixed seed.
#
# Not here, and why: B5 (Deaf-signer validation of 100 items) is a human study,
# not a GPU job. B4's literal form (redo the head reset on CSL-Daily or
# PHOENIX14T, where Uni-Sign works) needs video we do not have locally --
# data/external holds the checkpoints and no frames -- so its question is
# answered instead by the gloss->Chinese arm in run_rc5f.sh, which is a
# non-degenerate regime on this corpus.
#
# Resumable at cell granularity: every cell checks for its own finished output
# first, so a killed queue costs only the cell in flight. Run the snapshot, not
# this file -- bash re-reads a script by byte offset and editing one mid-run
# corrupts the parse.
#
#   cp run_rc5.sh .run_rc5.running.sh && nohup ./.run_rc5.running.sh \
#       > runs/logs/rc5_train.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Kevin cleared the card with chungche through the submission deadline, so there
# is no 20:00 stand-down this time. The bound is the deadline itself, and it is
# in code rather than in an estimate: PLAN_BY refuses to START a cell that cannot
# finish, HARD_STOP is the timeout every child gets.
PLAN_BY=${PLAN_BY:-2026-08-24 06:00}
HARD_STOP=${HARD_STOP:-2026-08-24 08:00}
CELL_BUDGET_MIN=${CELL_BUDGET_MIN:-45}
GPU_NEED_MIB=${GPU_NEED_MIB:-16000}
GPU_EVAL_MIB=${GPU_EVAL_MIB:-7000}
HOLD_FILE=${HOLD_FILE:-.gpu-hold}
CLIP_CONDS=${CLIP_CONDS:-local,blank_plain,swap_plain@0,swap_plain@1,swap_plain@2,shuffle_frames}
EPOCHS=${EPOCHS:-12}

PLAN_TS=$(date -d "$PLAN_BY" +%s) || exit 1
HARD_TS=$(date -d "$HARD_STOP" +%s) || exit 1
secs_to_hard() { echo $(( HARD_TS - $(date +%s) )); }
have_time()      { [ "$(( PLAN_TS - $(date +%s) ))" -ge $(( CELL_BUDGET_MIN * 60 )) ]; }
have_time_eval() { [ "$(( PLAN_TS - $(date +%s) ))" -ge 420 ]; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

wait_gpu() {
  local need=${1:-$GPU_NEED_MIB} waited=0
  while :; do
    # `.gpu-hold` is gpu_yield_watch.sh's flag that a foreign process is on the
    # card. Coordination with chungche covers the schedule, not the other twelve
    # accounts on this box.
    [ -f "$HOLD_FILE" ] || { [ "$(gpu_free)" -ge "$need" ] && return 0; }
    have_time || return 1
    [ $(( waited % 900 )) -eq 0 ] && echo "### waiting for ${need} MiB ($(date +%H:%M))"
    sleep 60; waited=$(( waited + 60 ))
  done
}

# CELLS: label|fold|init|init_parts|init_mt5|init_mt5_parts|seed|extra_train_args|evals
#   init_mt5 of "-" means no second checkpoint.
#   label is the run directory name; it must satisfy 19_bootstrap.py's rule that
#   a directory-derived cell name either equals the cfg-derived one or extends it
#   with an underscore suffix, or the analysis refuses to load the whole sweep.
#   evals: which readouts to take. `full` = all four, `lex` = dev CE + diagnostic.
CELLS=(
  # ---- block 1: the two flagged cells, seeds 1 and 2 (O-P1) -----------------
  "auto|0|how2sign|all|mt5_base|lm_head|1||full"
  "auto|0|how2sign|all|mt5_base|lm_head|2||full"
  "auto|0|csl_daily|all|mt5_base|lm_head|1||full"
  "auto|0|csl_daily|all|mt5_base|lm_head|2||full"
  # ---- block 2: the reset-vs-geometry control (O-P3) ------------------------
  "auto|0|how2sign|all|rand_head_nm|lm_head|0||full"
  "auto|0|csl_daily|all|rand_head_nm|lm_head|0||full"
  "auto|0|how2sign|all|rand_head|lm_head|0||full"
  # ---- block 3: is it only a per-script logit scale? (A-W2 / A-B3) ----------
  "auto|0|how2sign|all|how2sign_cjk|lm_head|0||full"
  "auto|0|how2sign|all|how2sign_cjkgl|lm_head|0||full"
  "auto|0|how2sign|all|how2sign_gl|lm_head|0||full"
  # ---- block 4: does the localisation replicate? (O-P2) --------------------
  "auto|0|openasl|all|mt5_base|lm_head|0||full"
  "auto|0|wlasl|all|mt5_base|lm_head|0||full"
  "auto|0|wlasl|all|csl_daily|lm_head|0||full"
  # ---- block 5: the learning-rate sweep (A-B1 / A-W6) ----------------------
  "f0_local_how2sign_lr1e4_full_k4_s0|0|how2sign|all|-|-|0|--lr-mt5 1e-4|full"
  "f0_local_how2sign_lr3e4_full_k4_s0|0|how2sign|all|-|-|0|--lr-mt5 3e-4|full"
  "f0_local_how2sign+mt5_base-lm_head_lr1e4_full_k4_s0|0|how2sign|all|mt5_base|lm_head|0|--lr-mt5 1e-4|lex"
  "f0_local_how2sign+mt5_base-lm_head_lr3e4_full_k4_s0|0|how2sign|all|mt5_base|lm_head|0|--lr-mt5 3e-4|lex"
  # ---- block 6: the strong written-language instrument (A-B2) --------------
  "auto|0|how2sign|all|gloss_lm|lm_head|0||full"
  "auto|0|csl_daily|pose|gloss_lm|lm_head|0||lex"
  # ---- block 7: model-selection sensitivity (A-B6) -------------------------
  "f0_local_how2sign_sl_full_k4_s0|0|how2sign|all|-|-|0|--save-last|lex"
  "f0_local_csl_daily_sl_full_k4_s0|0|csl_daily|all|-|-|0|--save-last|lex"
  "f0_local_how2sign+mt5_base-lm_head_sl_full_k4_s0|0|how2sign|all|mt5_base|lm_head|0|--save-last|lex"
)

FAILED=(); DEFERRED=(); SKIPPED=()

cell_done() {
  local name=$1 got
  [ -f "runs/$name/best.pt" ] && [ -f "runs/$name/log.json" ] || return 1
  got=$("$PY" -c "import json;print(json.load(open('runs/$name/log.json'))['epochs_done'])" 2>/dev/null) || return 1
  [ "${got:-0}" -ge "$EPOCHS" ]
}

run_eval() {
  local name=$1 out=$2 data=$3 conds=$4
  [ -f "runs/$name/$out" ] && return 0
  have_time_eval || { DEFERRED+=("eval $out $name"); return 1; }
  wait_gpu "$GPU_EVAL_MIB" || { DEFERRED+=("eval $out $name"); return 1; }
  timeout -k 60 "$(secs_to_hard)" "$PY" scripts/05_eval.py \
    --ckpt "runs/$name/best.pt" --data "$data" --batch 4 --workers 2 \
    --conditions "$conds" --out "runs/$name/$out" 2>&1 | grep -v "^Loading weights"
  [ "${PIPESTATUS[0]}" -eq 0 ] && [ -f "runs/$name/$out" ] && return 0
  rm -f "runs/$name/$out"; FAILED+=("eval $out $name"); return 1
}

echo "### rc5 training queue starts $(date '+%F %H:%M'), ${#CELLS[@]} cells,"
echo "### plan-by $PLAN_BY, hard stop $HARD_STOP"

for spec in "${CELLS[@]}"; do
  IFS='|' read -r label f init parts mt5 mt5parts seed xargs evals <<< "$spec"

  # Rebuild 04_train.py's own name when the cell does not need a tag, so the
  # directory and the cfg agree and 19_bootstrap.py can pool it.
  if [ "$label" = "auto" ]; then
    nm="$init"; [ "$parts" != "all" ] && nm="$nm-$parts"
    [ "$mt5" != "-" ] && nm="$nm+$mt5-$mt5parts"
    name="f${f}_local_${nm}_full_k4_s${seed}"
    tagarg=()
  else
    name="$label"
    tagarg=(--tag "$name")
  fi

  # Block 6 is gated on a donor another queue builds; skip rather than fail, so
  # a re-run picks it up once the file lands.
  if [ "$mt5" != "-" ]; then
    donor=$("$PY" -c "
import sys; sys.path.insert(0,'src')
from tsl.unisign import init_path
try: print(init_path('$mt5'))
except SystemExit: print('MISSING')
" 2>/dev/null)
    if [ "$donor" = "MISSING" ] || [ -z "$donor" ]; then
      echo "--- $name SKIPPED: donor '$mt5' not on disk yet"
      SKIPPED+=("$name (donor $mt5)")
      continue
    fi
  fi

  if cell_done "$name"; then
    echo "--- $name already trained"
  else
    have_time || { DEFERRED+=("$name"); continue; }
    wait_gpu  || { DEFERRED+=("$name"); continue; }
    echo "--- $name ($(date +%H:%M))"
    mt5args=()
    [ "$mt5" != "-" ] && mt5args=(--init-mt5 "$mt5" --init-mt5-parts "$mt5parts")
    # shellcheck disable=SC2086
    timeout -k 60 "$(secs_to_hard)" "$PY" scripts/04_train.py \
      --fold "$f" --system local --init "$init" --init-parts "$parts" \
      "${mt5args[@]}" "${tagarg[@]}" \
      --tune full --ctx-k 4 --epochs "$EPOCHS" --seed "$seed" $xargs
    rc=$?
    if [ "$rc" -ne 0 ]; then FAILED+=("train $name (exit $rc)"); continue; fi
  fi

  run_eval "$name" eval.json       data     local,text_only   || continue
  run_eval "$name" eval_lex.json   data/lex local,blank_plain || continue
  if [ "$evals" = "full" ]; then
    run_eval "$name" eval_blank.json data     blank_plain    || continue
    run_eval "$name" eval_clip.json  data/lex "$CLIP_CONDS"  || continue
  fi

  # B6's second readout: the same trajectory at its last epoch. Only the
  # --save-last cells have the file.
  if [ -f "runs/$name/last.pt" ] && [ ! -f "runs/$name/eval_lex_last.json" ]; then
    if have_time_eval && wait_gpu "$GPU_EVAL_MIB"; then
      echo "--- $name last-epoch readout ($(date +%H:%M))"
      timeout -k 60 "$(secs_to_hard)" "$PY" scripts/05_eval.py \
        --ckpt "runs/$name/last.pt" --data data/lex --batch 4 --workers 2 \
        --conditions local,blank_plain \
        --out "runs/$name/eval_lex_last.json" 2>&1 | grep -v "^Loading weights"
      [ "${PIPESTATUS[0]}" -eq 0 ] || FAILED+=("eval_lex_last $name")
    fi
  fi
done

echo
[ ${#SKIPPED[@]}  -ne 0 ] && { echo "### skipped (donor missing, re-run later):"; printf '  %s\n' "${SKIPPED[@]}"; }
[ ${#DEFERRED[@]} -ne 0 ] && { echo "### deferred (out of time):"; printf '  %s\n' "${DEFERRED[@]}"; }
[ ${#FAILED[@]}   -ne 0 ] && { echo "### failed:"; printf '  %s\n' "${FAILED[@]}"; exit 1; }
echo "### rc5 training queue complete ($(date '+%F %H:%M'))"
