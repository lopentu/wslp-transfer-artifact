#!/usr/bin/env bash
# Score every trained cell on the LEXICAL contrastive diagnostic (data/lex).
#
# Evaluation only. Nothing here trains, and nothing here writes to an existing
# eval.json: each run gets `eval_lex.json` alongside the pronoun-set `eval.json`,
# so the two diagnostics can be compared cell by cell and neither can overwrite
# the other. Re-running the script re-does only the cells that are missing.
#
# Why no retraining is needed: the contrastive task is a *readout*, not an
# objective. `04_train.py` never sees a candidate set — it optimises plain
# cross-entropy on the reference translation — and model selection is on dev
# cross-entropy, which is independent of any item set. Changing the items
# therefore changes only the question asked of a finished likelihood model.
#
#   ./run_lex_sweep.sh                 # the whole thing, priority order
#   PLAN_BY=18:00 ./run_lex_sweep.sh   # tighter deadline
#   DRY=1 ./run_lex_sweep.sh           # print the plan and exit
#
# ---------------------------------------------------------------- GPU sharing
#
# Three parties want this card and two of them are Kevin's. The colleague holds
# 21:00-09:00 by agreement; Kevin's PACLIC submission has first claim on the
# morning and is the more urgent of his two deadlines. So 09:00 is not "ours" —
# it is merely the end of somebody else's window, and taking the card on the
# stroke of nine would evict PACLIC.
#
# Hence the gate below is a *condition*, not a clock. After EARLIEST we sit out
# SETTLE_MIN without even looking, giving a PACLIC job time to claim the card
# first; then we require the card to be completely idle for QUIET_MIN
# consecutive minutes before taking it, so a PACLIC run that is merely between
# stages is not mistaken for a free card. `touch .gpu-hold` keeps us off
# indefinitely, and removing it lets the queue proceed.
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
DATA=${DATA:-data/lex}
OUT_NAME=${OUT_NAME:-eval_lex.json}
# `text_only` is deliberately absent. It is the one condition here that sits in
# CTX_TEXT, so it alone is scored under PROMPT_CTX while `local` and
# `blank_plain` get PROMPT_PLAIN — and the lexical distractors are rank-balanced
# under PROMPT_PLAIN, which puts a text-only LM at chance there and at 66% with
# context. Running all three would put a condition with an uncontrolled text
# shortcut in the same column as two without one. `blank_plain` is the clean
# one-factor ablation anyway (see tsl/conditions.py); what the preceding Chinese
# alone buys is reported from 07_shortcuts.py instead, and costs no GPU.
CONDITIONS=${CONDITIONS:-local,blank_plain}

# 11:00, not 09:00: PACLIC was confirmed on 8/14 to need the card from 09:00 for
# two to three hours. EARLIEST is the earliest we may *look*, not a start time —
# the idle test below still has to pass, so an overrunning PACLIC just delays us.
EARLIEST=${EARLIEST:-11:00}
SETTLE_MIN=${SETTLE_MIN:-20}
QUIET_MIN=${QUIET_MIN:-15}
HOLD_FILE=${HOLD_FILE:-.gpu-hold}
PLAN_BY=${PLAN_BY:-20:00}
HARD_STOP=${HARD_STOP:-20:30}
CELL_BUDGET_MIN=${CELL_BUDGET_MIN:-8}
GPU_NEED_MIB=${GPU_NEED_MIB:-9000}
DRY=${DRY:-}

[ -f "$DATA/records.jsonl" ] || { echo "### no $DATA/records.jsonl — run scripts/16_build_lexitems.py first"; exit 1; }
[ -e "$DATA/folds.json" ]   || { echo "### no $DATA/folds.json — the lexical set must use the SAME folds"; exit 1; }

PLAN_TS=$(date -d "today $PLAN_BY" +%s) || { echo "### unparseable PLAN_BY"; exit 1; }
HARD_TS=$(date -d "today $HARD_STOP" +%s) || { echo "### unparseable HARD_STOP"; exit 1; }
secs_to_plan() { echo $(( PLAN_TS - $(date +%s) )); }
secs_to_hard() { echo $(( HARD_TS - $(date +%s) )); }
have_time()    { [ "$(secs_to_plan)" -ge $(( CELL_BUDGET_MIN * 60 )) ]; }

# ------------------------------------------------------------- the cell list
#
# Priority order, because the deadline may cut the tail off. The fold-0
# factorial is the paper's headline and the only place the lexical set can
# confirm or overturn the encoder null result, so it goes first; the 6x3 main
# table second; seed reruns last, since they answer a question about variance
# that a partial answer still serves.
FACTORIAL=(
  f0_local_csl_daily_full_k4_s0
  f0_local_csl_stage1_full_k4_s0
  f0_local_how2sign_full_k4_s0
  f0_local_openasl_full_k4_s0
  f0_local_wlasl_full_k4_s0
  f0_local_random_full_k4_s0
  f0_local_how2sign-pose+csl_daily-mt5_full_k4_s0
  f0_local_wlasl-pose+csl_daily-mt5_full_k4_s0
  f0_local_csl_daily-pose+how2sign-mt5_full_k4_s0
  f0_local_csl_daily-pose+csl_stage1-mt5_full_k4_s0
  f0_local_csl_stage1-pose+csl_daily-mt5_full_k4_s0
  f0_local_csl_daily-pose_full_k4_s0
  f0_local_csl_daily-mt5_full_k4_s0
)
MAIN=(); SEEDS=()
for d in runs/f[0-2]_local_*_full_k4_s0; do
  n=$(basename "$d"); [ -f "$d/best.pt" ] || continue
  case " ${FACTORIAL[*]} " in *" $n "*) continue;; esac
  MAIN+=("$n")
done
for d in runs/f[0-2]_local_*_full_k4_s[12]; do
  n=$(basename "$d"); [ -f "$d/best.pt" ] && SEEDS+=("$n")
done

CELLS=("${FACTORIAL[@]}" "${MAIN[@]}" "${SEEDS[@]}")
TODO=()
for n in "${CELLS[@]}"; do
  [ -f "runs/$n/best.pt" ] || { echo "### skip $n (no best.pt)"; continue; }
  [ -f "runs/$n/$OUT_NAME" ] && continue
  TODO+=("$n")
done

echo "### eval sweep on $DATA: ${#TODO[@]} cells to do of ${#CELLS[@]} known"
echo "### data $DATA  conditions $CONDITIONS  -> runs/*/$OUT_NAME"
echo "### earliest $EARLIEST, then settle ${SETTLE_MIN}m and require ${QUIET_MIN}m idle; plan-by $PLAN_BY, hard stop $HARD_STOP"
if [ -n "$DRY" ]; then printf '  %s\n' "${TODO[@]}"; exit 0; fi
[ ${#TODO[@]} -eq 0 ] && { echo "### nothing to do"; exit 0; }

# ------------------------------------------------------------------ the gate
gpu_busy_procs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . ; }
gpu_free_mib()   { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

EARLIEST_TS=$(date -d "today $EARLIEST" +%s)
now=$(date +%s)
if [ "$now" -lt "$EARLIEST_TS" ]; then
  echo "### before $EARLIEST — sleeping $(( (EARLIEST_TS - now) / 60 )) min"
  sleep $(( EARLIEST_TS - now ))
fi
echo "### settling for ${SETTLE_MIN} min so PACLIC can claim the card first  ($(date +%H:%M))"
sleep $(( SETTLE_MIN * 60 ))

quiet=0
while :; do
  have_time || { echo "### plan-by reached while waiting for a clear card; nothing started"; exit 0; }
  if [ -f "$HOLD_FILE" ]; then
    echo "### $HOLD_FILE present — holding off  ($(date +%H:%M))"; quiet=0; sleep 300; continue
  fi
  procs=$(gpu_busy_procs); free=$(gpu_free_mib)
  if [ "${procs:-1}" -eq 0 ] && [ "${free:-0}" -ge "$GPU_NEED_MIB" ]; then
    quiet=$(( quiet + 1 ))
    [ "$quiet" -ge "$QUIET_MIN" ] && { echo "### card idle ${QUIET_MIN} min straight — taking it  ($(date +%H:%M))"; break; }
  else
    [ "$quiet" -gt 0 ] && echo "### card busy again after ${quiet} min quiet (${procs} procs, ${free} MiB free) — restarting the count"
    quiet=0
  fi
  sleep 60
done

# ------------------------------------------------------------------ the work
DONE=(); FAILED=(); SKIPPED=()
for n in "${TODO[@]}"; do
  if ! have_time; then SKIPPED+=("$n"); continue; fi
  if [ -f "$HOLD_FILE" ]; then echo "### $HOLD_FILE appeared — stopping at a clean boundary"; SKIPPED+=("$n"); continue; fi
  echo "### $n  ($(date +%H:%M), $(( $(secs_to_plan) / 60 )) min to plan-by)"
  # `timeout` bounded by HARD_STOP, not by a guess: a hung eval must not become
  # somebody else's problem at 21:00.
  if timeout -k 60 "$(secs_to_hard)" "$PY" scripts/05_eval.py \
       --ckpt "runs/$n/best.pt" --data "$DATA" \
       --conditions "$CONDITIONS" --out "runs/$n/$OUT_NAME"; then
    DONE+=("$n")
  else
    echo "### FAILED $n (exit $?)"; FAILED+=("$n")
    rm -f "runs/$n/$OUT_NAME"   # never leave a partial file that looks complete
  fi
done

echo
echo "### done ${#DONE[@]}, failed ${#FAILED[@]}, not reached ${#SKIPPED[@]}  ($(date +%H:%M))"
[ ${#FAILED[@]}  -gt 0 ] && printf '  failed:      %s\n' "${FAILED[@]}"
[ ${#SKIPPED[@]} -gt 0 ] && printf '  not reached: %s\n' "${SKIPPED[@]}"
echo "### re-run the same command to pick up what is missing"
