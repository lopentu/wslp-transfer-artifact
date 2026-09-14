#!/usr/bin/env bash
# The forward-pass half of the 8/22-review response: everything the reviews ask
# for that needs no training. Runs BESIDE run_rc5.sh rather than after it --
# inference needs ~7 GB and a training cell peaks at 14.4 GB on a 24 GB card, and
# these are the items both reviews rank first.
#
#   A  A-B2/B4  the gloss->Chinese arms (scripts/28_gloss_lm.py). First, because
#               run_rc5.sh's block 6 is gated on the donor this writes, and
#               because the other arms are the only non-degenerate regime
#               available on this corpus (B4's literal form needs CSL-Daily or
#               PHOENIX14T video we do not have).
#   B  A-A2     the post-hoc graft at three seeds. ReviewerA: "if you can only do
#               one thing, do this" -- the paper's adaptation-mediated conclusion
#               rests on a single-seed null with CI [-5.2, +2.4], which cannot
#               separate "no rescue" from "a 2-point rescue". Fold 0 already has
#               three seeds of every model involved, so this is pure rescoring.
#               Both directions, plus the compatible-head control at each seed.
#   C  A-A3/W2  the same tensor rescaled instead of replaced. If multiplying the
#               ASL head's CJK rows by 1.56 rescues the model, then "directional
#               specialisation" is really "per-script logit scale" and the
#               paper's mechanism paragraph is wrong. Swept over five factors
#               rather than only the calibrated one, plus the global-radius and
#               both-at-once variants, plus -- the cell the paper is missing --
#               mT5-base's own projection grafted POST-hoc, which is the neutral
#               head's version of the §3.3 null.
#   D  O-P0/A5  free decoding of the rescued cells. ReviewerO's single
#               highest-return item: the 23.9 -> 48.5 headline is a contrastive
#               diagnostic, and nobody knows whether it moves BLEU/chrF or output
#               diversity at all. Weights already exist; this is beam search.
#
# Synthetic-donor grafts are written to `eval_synthead_*.json`, NOT
# `eval_posthoc_*.json`: 19_bootstrap.py and make_numbers.py glob the latter and
# label a cell by its donor's PARENT DIRECTORY, so eleven donors that all live in
# data/external/head_donors would collapse into one mislabelled row.
#
#   cp run_rc5f.sh .run_rc5f.running.sh && nohup ./.run_rc5f.running.sh \
#       > runs/logs/rc5_fwd.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PLAN_BY=${PLAN_BY:-2026-08-24 06:00}
HARD_STOP=${HARD_STOP:-2026-08-24 08:00}
GPU_EVAL_MIB=${GPU_EVAL_MIB:-7000}
GPU_TRAIN_MIB=${GPU_TRAIN_MIB:-11000}
GLOSS_BATCH=${GLOSS_BATCH:-8}
HOLD_FILE=${HOLD_FILE:-.gpu-hold}
POST_CONDS=${POST_CONDS:-local,swap_plain@0,swap_plain@1,swap_plain@2}
DONORS=data/external/head_donors

PLAN_TS=$(date -d "$PLAN_BY" +%s) || exit 1
HARD_TS=$(date -d "$HARD_STOP" +%s) || exit 1
secs_to_hard() { echo $(( HARD_TS - $(date +%s) )); }
have_time() { [ "$(( PLAN_TS - $(date +%s) ))" -ge 420 ]; }
gpu_free()  { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

wait_gpu() {
  local need=${1:-$GPU_EVAL_MIB} waited=0
  while :; do
    [ -f "$HOLD_FILE" ] || { [ "$(gpu_free)" -ge "$need" ] && return 0; }
    have_time || return 1
    [ $(( waited % 900 )) -eq 0 ] && echo "### waiting for ${need} MiB ($(date +%H:%M))"
    sleep 60; waited=$(( waited + 60 ))
  done
}

FAILED=(); DEFERRED=()

# ---------------------------------------------------------------------- block B
# recipient|donor|outfile — donors are real runs, so these DO belong in
# eval_posthoc_*.json and flow into the existing bootstrap.
POSTHOC=(
  # the headline null, seeds 1 and 2 (seed 0 is in the paper)
  "f0_local_how2sign_full_k4_s1|f0_local_csl_daily_full_k4_s1|eval_posthoc_csldaily.json"
  "f0_local_how2sign_full_k4_s2|f0_local_csl_daily_full_k4_s2|eval_posthoc_csldaily.json"
  # the reverse direction, which is where the paper sees real damage (54.8->40.3)
  "f0_local_csl_daily_full_k4_s1|f0_local_how2sign_full_k4_s1|eval_posthoc_how2sign.json"
  "f0_local_csl_daily_full_k4_s2|f0_local_how2sign_full_k4_s2|eval_posthoc_how2sign.json"
  # grafting the RESCUED cell's own trained head back onto intact How2Sign
  "f0_local_how2sign_full_k4_s1|f0_local_how2sign+csl_daily-lm_head_full_k4_s1|eval_posthoc_rescued.json"
  "f0_local_how2sign_full_k4_s2|f0_local_how2sign+csl_daily-lm_head_full_k4_s2|eval_posthoc_rescued.json"
  # the compatible-head control at each seed: the graft machinery's own cost
  "f0_local_how2sign_full_k4_s1|f0_local_how2sign_full_k4_s2|eval_posthoc_selfseed.json"
  "f0_local_csl_daily_full_k4_s1|f0_local_csl_daily_full_k4_s2|eval_posthoc_selfseed.json"
  "f0_local_csl_daily_full_k4_s2|f0_local_csl_daily_full_k4_s0|eval_posthoc_selfseed.json"
  "f0_local_how2sign_full_k4_s2|f0_local_how2sign_full_k4_s0|eval_posthoc_selfseed.json"
)
for spec in "${POSTHOC[@]}"; do
  IFS='|' read -r rcp donor out <<< "$spec"
  [ -f "runs/$rcp/$out" ] && { echo "--- $out on $rcp already done"; continue; }
  [ -f "runs/$donor/best.pt" ] || { echo "### donor $donor missing"; FAILED+=("$donor"); continue; }
  wait_gpu || { DEFERRED+=("posthoc $rcp"); break; }
  echo "--- posthoc $rcp <- $donor  ($(date +%H:%M))"
  timeout -k 60 "$(secs_to_hard)" "$PY" scripts/05_eval.py \
    --ckpt "runs/$rcp/best.pt" --data data/lex --batch 4 --workers 2 \
    --conditions "$POST_CONDS" --posthoc-head "runs/$donor/best.pt" \
    --out "runs/$rcp/$out" 2>&1 | grep -v "^Loading weights"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { rm -f "runs/$rcp/$out"; FAILED+=("$out $rcp"); }
done

# ---------------------------------------------------------------------- block C
# recipient|donor .run.pth|outfile
SYNTH=(
  # The neutral head, post-hoc. The paper grafts CSL-Daily's fine-tuned
  # projection and gets nothing; mT5-base's is the projection that WORKS at
  # initialisation (47.9), so its post-hoc null is the one that matters.
  "f0_local_how2sign_full_k4_s0|mt5_base_pass|eval_synthead_mt5base.json"
  "f0_local_how2sign_full_k4_s1|mt5_base_pass|eval_synthead_mt5base.json"
  "f0_local_how2sign_full_k4_s2|mt5_base_pass|eval_synthead_mt5base.json"
  "f0_local_csl_daily_full_k4_s0|mt5_base_pass|eval_synthead_mt5base.json"
  # A-A3: the model's OWN head, rescaled. Nothing is replaced, so a rescue here
  # would mean the collapse was a scale artefact all along.
  "f0_local_how2sign_full_k4_s0|f0_local_how2sign_full_k4_s0_cjk1p2|eval_synthead_cjk1p2.json"
  "f0_local_how2sign_full_k4_s0|f0_local_how2sign_full_k4_s0_cjk1p6059|eval_synthead_cjk1p6.json"
  "f0_local_how2sign_full_k4_s0|f0_local_how2sign_full_k4_s0_cjk2p5|eval_synthead_cjk2p5.json"
  "f0_local_how2sign_full_k4_s0|f0_local_how2sign_full_k4_s0_cjk4|eval_synthead_cjk4.json"
  "f0_local_how2sign_full_k4_s0|f0_local_how2sign_full_k4_s0_gl|eval_synthead_gl.json"
  "f0_local_how2sign_full_k4_s0|f0_local_how2sign_full_k4_s0_cjkgl|eval_synthead_cjkgl.json"
  # The identity control for the whole block: the same head re-emitted through
  # the same code path. Anything other than "no change" means the synthetic-donor
  # machinery itself moves the number, and every row above is uninterpretable.
  "f0_local_how2sign_full_k4_s0|f0_local_how2sign_full_k4_s0_cjk1|eval_synthead_identity.json"
)
for spec in "${SYNTH[@]}"; do
  IFS='|' read -r rcp donor out <<< "$spec"
  [ -f "runs/$rcp/$out" ] && { echo "--- $out on $rcp already done"; continue; }
  [ -f "$DONORS/$donor.run.pth" ] || { echo "### synth donor $donor missing"; FAILED+=("$donor"); continue; }
  wait_gpu || { DEFERRED+=("synthead $rcp $donor"); break; }
  echo "--- synthead $rcp <- $donor  ($(date +%H:%M))"
  timeout -k 60 "$(secs_to_hard)" "$PY" scripts/05_eval.py \
    --ckpt "runs/$rcp/best.pt" --data data/lex --batch 4 --workers 2 \
    --conditions "$POST_CONDS" --posthoc-head "$DONORS/$donor.run.pth" \
    --out "runs/$rcp/$out" 2>&1 | grep -v "^Loading weights"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { rm -f "runs/$rcp/$out"; FAILED+=("$out $rcp"); }
done

# ---------------------------------------------------------------------- block A
# Ordered after B and C, not before them, for a memory reason: a training cell
# next door peaks at 14.4 GB of the 24 GB card, and fine-tuning a full mT5 with
# AdamW does not fit in the ~8.7 GB left, where a 7 GB scoring pass does. B and C
# are also ~1 h in total, so the donor still lands hours before run_rc5.sh's
# block 6 reaches it.
GLOSS=("mt5_base" "how2sign" "csl_daily" "rand_head_nm" "openasl")
for h in "${GLOSS[@]}"; do
  tag=""; [ "$h" != "mt5_base" ] && tag="_$h"
  rep="data/gloss_lm${tag}.json"
  [ -f "$rep" ] && { echo "--- gloss_lm $h already done"; continue; }
  wait_gpu "$GPU_TRAIN_MIB" || { DEFERRED+=("gloss_lm $h"); break; }
  echo "--- gloss_lm --init-head $h  ($(date +%H:%M))"
  timeout -k 60 "$(secs_to_hard)" "$PY" scripts/28_gloss_lm.py \
    --fold 0 --init-head "$h" --batch "$GLOSS_BATCH" 2>&1 | grep -v "^Loading weights"
  [ "${PIPESTATUS[0]}" -eq 0 ] || FAILED+=("gloss_lm $h")
done

# ---------------------------------------------------------------------- block D
# Beam search over the 1,760-utterance test split, so this is the expensive block
# and it goes last. run|posthoc donor ("-" for none)|outfile
GEN=(
  "f0_local_how2sign+mt5_base-lm_head_full_k4_s0|-|eval_gen.json"
  "f0_local_how2sign+csl_daily-lm_head_full_k4_s0|-|eval_gen.json"
  "f0_local_csl_daily+mt5_base-lm_head_full_k4_s0|-|eval_gen.json"
  "f0_local_how2sign_full_k4_s0|f0_local_csl_daily_full_k4_s0|eval_gen_posthoc.json"
  "f0_local_how2sign+csl_daily-mt5_nohead_full_k4_s0|-|eval_gen.json"
  "f0_local_how2sign+rand_head_nm-lm_head_full_k4_s0|-|eval_gen.json"
  "f0_local_csl_daily+rand_head_nm-lm_head_full_k4_s0|-|eval_gen.json"
  "f1_local_how2sign+mt5_base-lm_head_full_k4_s0|-|eval_gen.json"
  "f1_local_csl_daily+mt5_base-lm_head_full_k4_s0|-|eval_gen.json"
  "f2_local_how2sign+mt5_base-lm_head_full_k4_s0|-|eval_gen.json"
  "f2_local_csl_daily+mt5_base-lm_head_full_k4_s0|-|eval_gen.json"
)
for spec in "${GEN[@]}"; do
  IFS='|' read -r run donor out <<< "$spec"
  [ -f "runs/$run/$out" ] && { echo "--- gen $out on $run already done"; continue; }
  [ -f "runs/$run/best.pt" ] || { echo "--- gen $run not trained yet, skipping"; DEFERRED+=("gen $run"); continue; }
  post=()
  if [ "$donor" != "-" ]; then
    [ -f "runs/$donor/best.pt" ] || { FAILED+=("gen donor $donor"); continue; }
    post=(--posthoc-head "runs/$donor/best.pt")
  fi
  wait_gpu || { DEFERRED+=("gen $run"); break; }
  echo "--- gen $run ${donor/-/} ($(date +%H:%M))"
  timeout -k 60 "$(secs_to_hard)" "$PY" scripts/05_eval.py \
    --ckpt "runs/$run/best.pt" --data data --batch 4 --workers 2 \
    --conditions local --generate --gen-conditions local "${post[@]}" \
    --out "runs/$run/$out" 2>&1 | grep -v "^Loading weights"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { rm -f "runs/$run/$out"; FAILED+=("gen $run"); }
done

echo
[ ${#DEFERRED[@]} -ne 0 ] && { echo "### deferred:"; printf '  %s\n' "${DEFERRED[@]}"; }
[ ${#FAILED[@]}   -ne 0 ] && { echo "### failed:"; printf '  %s\n' "${FAILED[@]}"; exit 1; }
echo "### rc5f complete ($(date '+%F %H:%M'))"
