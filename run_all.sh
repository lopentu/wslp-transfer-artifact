#!/usr/bin/env bash
# The whole experiment, in dependency order. Each stage is skippable and
# resumable; feature extraction and training both skip work already on disk.
#
#   ./run_all.sh gate       # day 1: audit + shortcut baselines, NO GPU training
#   ./run_all.sh features   # one-time visual feature precompute
#   ./run_all.sh main       # the transfer table: 6 initialisations x 3 folds
#   ./run_all.sh gen        # BLEU/chrF for whatever is trained; re-runnable
#   ./run_all.sh context    # axis 2, one fold: does discourse context help at all
#   ./run_all.sh ablate     # from-scratch encoder, only if time allows
#   ./run_all.sh report
#   ./run_all.sh all
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate

# Peak allocation is ~13.4 GB against another tenant's ~6.5 GB on a 24.5 GB card,
# so the margin is thin enough that fragmentation alone can decide whether a run
# survives. Expandable segments give the allocator room to grow a block instead
# of failing between two free ones.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

FOLDS=${FOLDS:-3}
SEED=${SEED:-0}
LLM=${LLM:-Qwen/Qwen2.5-1.5B-Instruct}
WORKERS=${WORKERS:-12}
# The transfer axis — the paper's main result. Ordered so that the rows that pull
# "same written language" apart from "same sign language" land first.
INITS=${INITS:-csl_daily how2sign random csl_stage1 openasl wlasl}
CTXK=${CTXK:-4}
EPOCHS=${EPOCHS:-12}

# `local` is the whole main sweep. With no context on either path it reproduces
# upstream Uni-Sign's own forward exactly, so the transfer comparison is between
# initialisations and nothing else. The context axis (README §4, axis 2) is a
# separate question and a separate sweep — see stage_context.
SYSTEMS=${SYSTEMS:-local}

# Full fine-tuning, not LoRA, and this is not a preference. LoRA targets the mT5
# attention and feed-forward blocks but not `lm_head` or the embeddings, so a
# checkpoint whose decoder was fine-tuned to emit English cannot move its output
# distribution to Chinese: `how2sign` sat at dev 9.30 for five epochs, having
# barely moved from its untrained 9.99, while `csl_daily` — which already emitted
# Chinese — reached 5.99. That gap is adapter placement, not transfer. Under full
# fine-tuning the same pair is 6.79 against 6.10 at a matched two epochs. Every
# row must be able to reach the output language or the table measures the wrong
# thing.
TUNE=${TUNE:-full}

# Contrastive scoring only, on the two conditions a no-context model has an
# interpretation for: `local`, and `text_only` as the control that blanks the
# target clip. Free decoding for BLEU/chrF is a separate stage because it costs
# more than training does and can be re-run from `best.pt` at any time, whereas
# the GPU window for training cannot.
EVAL_CONDS=${EVAL_CONDS:-local,text_only}
GEN=${GEN:-0}

stage_gate() {
  echo "### audit"
  python3 scripts/00_audit.py | tee data/audit.txt
  echo "### build dataset"
  python3 scripts/01_build_dataset.py --folds "$FOLDS"
  echo "### verify the dialogue L/R mapping before anything depends on it"
  python3 scripts/03_extract_features.py --check-sides
  echo "### shortcut baselines — the go/no-go gate"
  python3 scripts/07_shortcuts.py --llm "$LLM"
}

stage_features() {
  echo "### pose features (one-time, ~1-3 h on 32 cores)"
  python3 scripts/03_extract_features.py --backend rtmpose --device cpu \
    --workers 6 --threads 2
  echo "### reconcile the store against records.jsonl — non-zero means do NOT train"
  python3 scripts/11_verify_features.py
}

# One cell of the design: train it if it is not already trained, evaluate it if
# it is not already evaluated. Both halves are skipped when their output exists,
# so an evicted sweep is restarted by re-running the same command.
#
# A failing cell does NOT stop the sweep. The card is shared and a returning
# tenant shows up as an OOM in whichever run happens to be on the GPU at the
# time; losing the remaining twelve hours of queue to it would be the expensive
# failure. Failures are collected and re-listed at the end, and the exit status
# is non-zero, so nothing is lost quietly either.

# Full fine-tuning peaks at 14.4 GB on a 24.5 GB card that someone else is also
# using, and their job comes and goes: it was 18.4 GB early on 8/11, gone by
# 04:00, back at 23.3 GB by 22:30. So wait for a window before each cell instead
# of starting into a full card and losing the cell to an OOM. A returning tenant
# then pauses the sweep rather than ending it.
GPU_NEED_MIB=${GPU_NEED_MIB:-16000}
GPU_WAIT_MAX=${GPU_WAIT_MAX:-43200}   # 12 h, then give up and say so

# Do not take the card before this local time, even if it frees earlier. The
# card is shared with people, not just with jobs: the small hours are when the
# other tenants' work lands, and grabbing the first window at 01:00 takes it from
# someone who is waiting for it. Empty means start as soon as there is room.
#   START_AFTER=04:00 ./run_all.sh main
START_AFTER=${START_AFTER:-}
hold_until() {
  [ -n "$1" ] || return 0
  local target remain
  if ! target=$(date -d "today $1" +%s 2>/dev/null); then
    echo "### START_AFTER=$1 is not a time I can parse; starting now"
    return 0
  fi
  if [ "$target" -le "$(date +%s)" ]; then target=$(date -d "tomorrow $1" +%s); fi
  while :; do
    remain=$((target - $(date +%s)))
    if [ "$remain" -le 0 ]; then break; fi
    echo "### holding until $1 before touching the GPU: $((remain / 60)) min to go  ($(date +%H:%M))"
    if [ "$remain" -lt 1800 ]; then sleep "$remain"; else sleep 1800; fi
  done
  echo "### $1 reached ($(date +%H:%M)) — looking for a window"
}
wait_for_gpu() {
  local waited=0 free
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge "$GPU_NEED_MIB" ]; then
      if [ "$waited" -gt 0 ]; then
        echo "### GPU free (${free} MiB) after $((waited / 60)) min of waiting"
      fi
      return 0
    fi
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

FAILED=()
cell() {
  local f=$1 sys=$2 init=$3
  local name="f${f}_${sys}_${init}_${TUNE}_k${CTXK}_s${SEED}"
  if [ -f "runs/$name/best.pt" ]; then
    echo "--- $name already trained"
  else
    wait_for_gpu || { FAILED+=("gpu-window $name"); return; }
    echo "--- $name  ($(date +%H:%M))"
    python3 scripts/04_train.py --fold "$f" --system "$sys" --init "$init" \
      --tune "$TUNE" --ctx-k "$CTXK" --epochs "$EPOCHS" --seed "$SEED" \
      || { FAILED+=("train $name"); return; }
  fi
  [ -f "runs/$name/eval.json" ] && return 0
  local ev=(--ckpt "runs/$name/best.pt" --conditions "$EVAL_CONDS")
  if [ "$GEN" = 1 ]; then ev+=(--generate --gen-conditions local); fi
  python3 scripts/05_eval.py "${ev[@]}" || FAILED+=("eval $name")
}

report_failures() {
  [ ${#FAILED[@]} -eq 0 ] && return 0
  echo "### ${#FAILED[@]} cell(s) failed — rerun the same command to retry just these:"
  printf '  %s\n' "${FAILED[@]}"
  return 1
}

stage_main() {
  hold_until "$START_AFTER"
  # Every initialisation on fold 0 before any of it is repeated over folds: a
  # sweep cut off half way should leave one complete transfer table behind
  # rather than one complete fold of an incomplete table.
  for f in $(seq 0 $((FOLDS - 1))); do
    for init in $INITS; do
      for sys in $SYSTEMS; do cell "$f" "$sys" "$init"; done
    done
  done
  report_failures
}

stage_split() {
  # Encoder credit against decoder credit. Every released checkpoint bundles an
  # encoder that has seen sign language with a decoder narrowed to one written
  # language, so no comparison between whole checkpoints can say which half a
  # difference came from — and that is the paper's actual claim. `--init-parts`
  # takes one half and leaves the other at its published initialisation.
  local init=${1:-csl_daily} f=${2:-0}
  for parts in pose mt5; do
    local name="f${f}_local_${init}-${parts}_${TUNE}_k${CTXK}_s${SEED}"
    if [ -f "runs/$name/best.pt" ]; then
      echo "--- $name already trained"
    else
      wait_for_gpu || { FAILED+=("gpu-window $name"); continue; }
      echo "--- $name  ($(date +%H:%M))"
      python3 scripts/04_train.py --fold "$f" --system local --init "$init" \
        --init-parts "$parts" --tune "$TUNE" --ctx-k "$CTXK" --epochs "$EPOCHS" \
        --seed "$SEED" || { FAILED+=("train $name"); continue; }
    fi
    [ -f "runs/$name/eval.json" ] || python3 scripts/05_eval.py \
      --ckpt "runs/$name/best.pt" --conditions "$EVAL_CONDS" || FAILED+=("eval $name")
  done
  report_failures
}

stage_gen() {
  # BLEU/chrF for the rows already trained. Separate from `main` because beam
  # search over the full test split is the expensive half of evaluation and
  # nothing else waits on it.
  for run in runs/*/best.pt; do
    d=$(dirname "$run")
    [ -f "$d/eval_gen.json" ] || python3 scripts/05_eval.py --ckpt "$run" \
      --conditions local --generate --gen-conditions local --out "$d/eval_gen.json"
  done
}

stage_context() {
  # Axis 2, and a separate question from the transfer table: does discourse
  # context help at all? One initialisation, one fold, both systems — enough to
  # answer it, and it is not the paper's main claim.
  for sys in context local; do cell 0 "$sys" csl_daily; done
  report_failures
}

stage_ablate() {
  # Only after the main result is in hand. The from-scratch encoder: no
  # sign-language pretraining anywhere in the stack, unlike `--init random`,
  # which still keeps Uni-Sign's architecture and mT5.
  python3 scripts/04_train.py --fold 0 --system local --arch scratch \
    --backend rtmpose --seed "$SEED" --llm "$LLM"
  python3 scripts/05_eval.py --conditions local \
    --ckpt "runs/f0_local_rtmpose_none_k${CTXK}_s${SEED}/best.pt"
}

stage_report() {
  # Axis 3 needs the difficulty strata; they are CPU-only and take seconds, so
  # build them here rather than making the report depend on remembering to.
  [ -f data/strata.json ] || python3 scripts/12_strata.py
  python3 scripts/06_report.py runs/*/eval.json --latex paper_tables.tex --out results.txt
  cat results.txt
}

case "${1:-all}" in
  gate) stage_gate ;;
  features) stage_features ;;
  main) stage_main ;;
  split) shift; stage_split "$@" ;;
  gen) stage_gen ;;
  context) stage_context ;;
  ablate) stage_ablate ;;
  report) stage_report ;;
  all) stage_gate; stage_features; stage_main; stage_gen; stage_report ;;
  *) echo "usage: $0 {gate|features|main|gen|context|ablate|report|all}"; exit 1 ;;
esac
