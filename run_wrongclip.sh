#!/usr/bin/env bash
# The wrong-clip control: is `local` - `blank_plain` really "what the target clip
# is worth", or partly "how the network reacts to a degenerate input"?
#
# ---------------------------------------------------------------- the question
#
# Every Delta in the paper is `local` minus `blank_plain`, and blanking zeroes the
# 133 keypoints. No training clip ever looked like that, so a reviewer can read
# the ASL rows' *negative* Delta (-4.3, -6.0, significant) not as "the clip is
# interference" but as "an out-of-distribution input perturbs the model in a
# direction that happens to help". That objection is fatal to the strongest claim
# in the paper -- the collapsed models do not read the clip -- because it attacks
# the one measurement that claim rests on.
#
# So we keep the input in distribution and break only the correspondence:
#
#   swap_plain      a real clip of a DIFFERENT utterance, different paragraph,
#                   duration-matched to within a couple of milliseconds, drawn
#                   from this fold's own test split. Real signer, real motion,
#                   wrong sentence. Three independent draws (@0/@1/@2), so the
#                   figure is not one lucky assignment.
#   shuffle_frames  this clip's own frames, permuted. Same frame distribution as
#                   `local` exactly; temporal structure alone removed.
#
# What each outcome would mean, written down before the sweep ran:
#
#   local >> swap ~ blank   the model reads clip-specific information, and the
#                           blank ablation was measuring what it claimed to.
#   local ~ swap < blank    the model ignores the clip, and the negative Delta is
#                           an artefact of the degenerate input, not evidence that
#                           the clip interferes.
#   local ~ swap ~ blank    the model ignores the clip, full stop.
#
# Pilot on fold 0 (8/18, before this script existed), lexical set:
#   CSL-Daily  local 54.8  blank 35.2  swap 34.2  shuffle 36.7
#   How2Sign   local 24.5  blank 27.0  swap 24.2  shuffle 25.5
# i.e. the first row for CSL and the second for ASL. The sweep is what turns two
# fold-0 cells into a table.
#
# Evaluation only -- nothing here trains. The contrastive task is a readout over a
# finished likelihood model (`04_train.py` never sees a candidate set), so a new
# condition is a new question asked of the same weights.
#
# Output goes to `eval_clip.json`, NOT into `eval_lex.json`: every number now in
# the paper is generated from the existing files, and a sweep that rewrote them
# would silently move published cells. The new file also carries two fields the
# old ones lack -- `pick` (which of the four candidates, not merely whether it was
# gold) and `scores` -- which is what the inter-model agreement analysis needs.
#
#   nohup ./run_wrongclip.sh > runs/logs/wrongclip.log 2>&1 &
#   DRY=1 ./run_wrongclip.sh              # print the plan and exit
#   PRON=1 ./run_wrongclip.sh             # the referential set instead
# ---------------------------------------------------------------- GPU sharing
#
# CHANGED 8/19: the colleague has released the card through the submission, so
# there is no nightly hand-back and no 21:00 boundary until 2026-08-21 20:00.
# The deadline below is therefore an ABSOLUTE timestamp rather than "today
# 20:00", and EARLIEST is empty so the queue never parks waiting for a morning
# that has already been granted. Two consequences worth knowing:
#
#   * `date -d "$PLAN_BY"`, not `date -d "today $PLAN_BY"` -- the latter cannot
#     parse a full date, and would have silently produced an unparseable
#     deadline, which these scripts treat as a refusal to run.
#   * a queue that runs past midnight used to sleep until the next 09:00,
#     because `today 09:00` is in the future at 00:30. With EARLIEST empty the
#     hold is skipped entirely.
#
# The free-memory gates stay. They are what stops us walking into somebody
# else's job, and that risk did not go away with the schedule.
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
PRON=${PRON:-}
if [ -n "$PRON" ]; then
  DATA=${DATA:-data}; OUT_NAME=${OUT_NAME:-eval_clip_pron.json}
else
  DATA=${DATA:-data/lex}; OUT_NAME=${OUT_NAME:-eval_clip.json}
fi
CONDITIONS=${CONDITIONS:-local,blank_plain,swap_plain@0,swap_plain@1,swap_plain@2,shuffle_frames}

# The card is shared: the colleague holds it 21:00-09:00 by agreement, so nothing
# may start that cannot finish by PLAN_BY, and HARD_STOP is enforced by `timeout`
# rather than by an estimate of how long a cell takes.
EARLIEST=${EARLIEST:-}
HOLD_FILE=${HOLD_FILE:-.gpu-hold}
PLAN_BY=${PLAN_BY:-2026-08-21 20:00}
HARD_STOP=${HARD_STOP:-2026-08-21 20:45}
CELL_BUDGET_MIN=${CELL_BUDGET_MIN:-5}
# 8/18: raised from 7000, and BATCH/WORKERS added, after three cells were killed
# by CUDA OOM while the training sweep held the card. A training cell peaks at
# 14.6 GB of the 24.5 GB available, so an evaluation that peaks at 6.8 GB (batch
# 8) leaves ~3 GB of headroom, and allocator fragmentation eats that. Batch 4
# roughly halves the evaluation's peak and costs a minute a cell; the gate then
# refuses to start one at all unless there is genuine room. Two dataloader
# workers rather than four for the same reason on the host side: this box has
# 31 GB and the training job's workers are already resident.
# 8000, not 9500: with a training cell resident nvidia-smi reports ~8.6 GB free
# (the caching allocator holds more than the job's peak), so a 9500 gate parks
# this sweep for the whole 30 minutes of every training cell and it never runs.
# At BATCH=4 an evaluation peaks at roughly 4.6 GB -- 2.4 GB of fp32 weights plus
# activations -- so 8000 leaves over 3 GB of headroom, against the 1.8 GB that
# was not enough at BATCH=8.
GPU_NEED_MIB=${GPU_NEED_MIB:-8000}
# 8/19: the referential set is an auxiliary reading in the third draft -- one
# column of Table 1 and a short subsection -- so scoring all 92 cells on it buys
# numbers the paper has no place to put, at 7-8 min a cell (the referential items
# are roughly four times the cost of a lexical one) for about eleven hours of
# card time we would rather spend on the training queue. SCOPE=main restricts the
# sweep to MAIN: the six released initialisations on all three folds, which is
# exactly the set the lexical-vs-referential contrast is stated over. The other
# tiers stay in the file because the question they answer is real; it is just not
# this deadline's.
SCOPE=${SCOPE:-all}
BATCH=${BATCH:-4}
WORKERS=${WORKERS:-2}
DRY=${DRY:-}

[ -f "$DATA/records.jsonl" ] || { echo "### no $DATA/records.jsonl"; exit 1; }

PLAN_TS=$(date -d "$PLAN_BY" +%s) || { echo "### unparseable PLAN_BY"; exit 1; }
HARD_TS=$(date -d "$HARD_STOP" +%s) || { echo "### unparseable HARD_STOP"; exit 1; }
secs_to_plan() { echo $(( PLAN_TS - $(date +%s) )); }
secs_to_hard() { echo $(( HARD_TS - $(date +%s) )); }
have_time()    { [ "$(secs_to_plan)" -ge $(( CELL_BUDGET_MIN * 60 )) ]; }

# ------------------------------------------------------------- the cell list
#
# Priority order, because the window may cut the tail off. Rows the body's claims
# actually rest on come first: the six initialisations of Table 1 on all three
# folds (the negative-Delta claim is about the three ASL rows there), then the
# mixed-half cells of Table 2 whose Delta column carries the "reaches its score
# without reading the clip" sentence, then the one-half controls.
MAIN=()
for f in 0 1 2; do for i in csl_daily csl_stage1 random how2sign openasl wlasl; do
  MAIN+=("f${f}_local_${i}_full_k4_s0")
done; done
MIXED=()
for f in 0 1 2; do for c in how2sign-pose+csl_daily-mt5 wlasl-pose+csl_daily-mt5 \
                            csl_stage1-pose+csl_daily-mt5 csl_daily-pose+csl_stage1-mt5 \
                            csl_daily-pose+how2sign-mt5; do
  MIXED+=("f${f}_local_${c}_full_k4_s0")
done; done
HALVES=()
for f in 0 1 2; do for c in csl_daily-pose csl_daily-mt5; do
  HALVES+=("f${f}_local_${c}_full_k4_s0")
done; done
# Seeds and the encoder-only excursion rows last: they answer variance questions
# that a partial answer still serves.
# 8/19, after the body pass: block (b) has seven rows and MIXED covers five. The
# two missing ones are the OpenASL pairings, and they are the direct replications
# of the two claims that landed today -- csl_daily-pose+openasl-mt5 asks whether a
# SECOND English decoder also stops the model reading the clip, and
# openasl-pose+csl_daily-mt5 whether a SECOND ASL encoder pays the same
# set-dependent cost. Six cells, and they finish block (b) on both instruments.
REPL=()
for f in 0 1 2; do for c in csl_daily-pose+openasl-mt5 openasl-pose+csl_daily-mt5; do
  REPL+=("f${f}_local_${c}_full_k4_s0")
done; done
# 8/23. The output-projection cells were never scored on the referential set,
# because on 8/19 they did not exist. They carry findings 0d-0f now, and every
# claim in them is stated on the content-word instrument only -- including the
# abolition result, whose whole force is that a cell's clip dependence went to
# zero. "Zero on the lexical set" and "zero on both sets" are different claims,
# and A-Q5 turns on the two instruments disagreeing, so the referential reading
# of exactly these five cells is worth ~40 min of card time.
HEADS=()
for c in csl_daily+how2sign-lm_head how2sign+mt5_base-lm_head \
         csl_daily+mt5_base-lm_head how2sign+csl_daily-lm_head \
         how2sign+csl_daily-mt5_nohead; do
  HEADS+=("f0_local_${c}_full_k4_s0")
done
EXTRA=()
for d in runs/f[0-2]_local_*_full_k4_s[0-2] runs/*_e2[04]; do
  n=$(basename "$d"); [ -f "$d/best.pt" ] || continue
  case " ${MAIN[*]} ${MIXED[*]} ${HALVES[*]} ${HEADS[*]} " in *" $n "*) continue;; esac
  EXTRA+=("$n")
done

case "$SCOPE" in
  main) CELLS=("${MAIN[@]}") ;;
  # 8/19, after reading fold 0: the referential wrong-clip result rewrites §3.5,
  # and §3.5 states its contrast over the RE-PAIRED cells (block b), which hold
  # the mT5 half fixed so the encoder step is the only thing varying. MAIN alone
  # cannot express that sentence -- the released checkpoints vary both halves at
  # once. `body` is MAIN plus MIXED: exactly the cells the body's claims are
  # stated over, and nothing else.
  body) CELLS=("${MAIN[@]}" "${MIXED[@]}" "${REPL[@]}") ;;
  repl) CELLS=("${REPL[@]}") ;;
  heads) CELLS=("${HEADS[@]}") ;;
  all)  CELLS=("${MAIN[@]}" "${MIXED[@]}" "${REPL[@]}" "${HEADS[@]}" "${HALVES[@]}" "${EXTRA[@]}") ;;
  *)    echo "### SCOPE='$SCOPE' unknown (main|body|repl|heads|all)"; exit 1 ;;
esac
# `best.pt` is rewritten every epoch (04_train.py), so it does not mean the cell
# finished; `log.json`'s `epochs_done` does. Scoring a half-trained cell would put
# a row in the table at whatever epoch the run died on.
cell_done() {
  local n=$1 want=12 got
  case "$n" in *_e24) want=24;; *_e20) want=20;; esac
  [ -f "runs/$n/best.pt" ] && [ -f "runs/$n/log.json" ] || return 1
  got=$("$PY" -c "import json;print(json.load(open('runs/$n/log.json'))['epochs_done'])" 2>/dev/null) || return 1
  [ "${got:-0}" -ge "$want" ]
}

TODO=()
for n in "${CELLS[@]}"; do
  cell_done "$n" || { echo "### skip $n (not trained, or trained short)"; continue; }
  [ -f "runs/$n/$OUT_NAME" ] && continue
  TODO+=("$n")
done

echo "### wrong-clip sweep on $DATA -> runs/*/$OUT_NAME"
echo "### conditions $CONDITIONS"
echo "### ${#TODO[@]} cells to do of ${#CELLS[@]} known; plan-by $PLAN_BY, hard stop $HARD_STOP"
if [ -n "$DRY" ]; then printf '  %s\n' "${TODO[@]}"; exit 0; fi
[ ${#TODO[@]} -eq 0 ] && { echo "### nothing to do"; exit 0; }

gpu_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }
wait_for_gpu() {
  local waited=0 free
  while :; do
    [ -f "$HOLD_FILE" ] || {
      free=$(gpu_free_mib)
      [ "${free:-0}" -ge "$GPU_NEED_MIB" ] && return 0
    }
    have_time || return 1
    [ $(( waited % 900 )) -eq 0 ] && echo "### waiting for ${GPU_NEED_MIB} MiB  ($(date +%H:%M))"
    sleep 60; waited=$(( waited + 60 ))
  done
}

DONE=(); FAILED=(); SKIPPED=()
for n in "${TODO[@]}"; do
  have_time   || { SKIPPED+=("$n"); continue; }
  wait_for_gpu || { SKIPPED+=("$n"); continue; }
  echo "### $n  ($(date +%H:%M), $(( $(secs_to_plan) / 60 )) min to plan-by)"
  # PIPESTATUS, not the pipeline's status: the pipeline's is `grep`'s, so a
  # python that died would be reported as a success whenever grep still had a
  # line to print, and a python that printed nothing but succeeded would be
  # reported as a failure. Neither is acceptable in a loop that deletes output
  # files on failure.
  timeout -k 60 "$(secs_to_hard)" "$PY" scripts/05_eval.py \
       --ckpt "runs/$n/best.pt" --data "$DATA" --batch "$BATCH" \
       --workers "$WORKERS" --conditions "$CONDITIONS" \
       --out "runs/$n/$OUT_NAME" 2>&1 | grep -v "^Loading weights"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ] && [ -f "runs/$n/$OUT_NAME" ]; then
    DONE+=("$n")
  else
    echo "### FAILED $n (exit $rc)"; FAILED+=("$n")
    rm -f "runs/$n/$OUT_NAME"   # never leave a partial file that looks complete
  fi
done

echo
echo "### done ${#DONE[@]}, failed ${#FAILED[@]}, not reached ${#SKIPPED[@]}  ($(date +%H:%M))"
[ ${#FAILED[@]}  -gt 0 ] && printf '  failed:      %s\n' "${FAILED[@]}"
[ ${#SKIPPED[@]} -gt 0 ] && printf '  not reached: %s\n' "${SKIPPED[@]}"
echo "### re-run the same command to pick up what is missing"
exit 0
