#!/usr/bin/env bash
# The two mechanism probes, which need new measurements rather than new cells.
#
#   A-A4  scripts/29_interface_align.py — the objection the post-hoc null does
#         not yet answer. Grafting CSL-Daily's projection onto a fine-tuned
#         How2Sign model changes nothing, and the paper reads that as "there was
#         nothing to read". The alternative is "there was something to read, in a
#         basis the donor projection does not expect", and it survives everything
#         measured so far. Fit an interface map on dev and try again. The ladder
#         matters: a null after an ORTHOGONAL map is weak, a null after a map
#         trained directly on the donor head's own cross-entropy is decisive.
#
#   O-P4  scripts/30_grad_probe.py — the mechanism the paper asserts and has
#         never measured. Everything supporting "a mismatched projection stops
#         the visual representation from forming" is an endpoint after twelve
#         epochs. At step 0 the gradient reaching the visual branch can be read
#         directly, under each output projection, on the same batches.
#
# Ordering is a memory constraint, not a logical one. The interface probe holds
# one model at a time (~7 GB) so it can run beside a training cell; the gradient
# probe backpropagates through the full 582M mT5 and cannot, so it waits for the
# training queue to drain. Both bracket-trick their pgrep patterns and name the
# other queues' snapshots exactly -- `[r]un_rc5` would match this script's own
# command line and it would wait on itself forever, which is how 8/22 lost an
# afternoon.
#
#   cp run_rc5g.sh .run_rc5g.running.sh && nohup ./.run_rc5g.running.sh \
#       > runs/logs/rc5_probe.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PLAN_BY=${PLAN_BY:-2026-08-24 06:00}
HARD_STOP=${HARD_STOP:-2026-08-24 08:00}
GPU_EVAL_MIB=${GPU_EVAL_MIB:-7500}
GPU_GRAD_MIB=${GPU_GRAD_MIB:-13000}
HOLD_FILE=${HOLD_FILE:-.gpu-hold}
POST_CONDS=${POST_CONDS:-local,swap_plain@0,swap_plain@1,swap_plain@2}
DONORS=data/external/head_donors
WAIT_FWD=${WAIT_FWD:-[r]un_rc5f[.]running}
WAIT_TRAIN=${WAIT_TRAIN:-[r]un_rc5[.]running}

PLAN_TS=$(date -d "$PLAN_BY" +%s) || exit 1
HARD_TS=$(date -d "$HARD_STOP" +%s) || exit 1
secs_to_hard() { echo $(( HARD_TS - $(date +%s) )); }
have_time() { [ "$(( PLAN_TS - $(date +%s) ))" -ge 600 ]; }
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

wait_queue() {
  local pat=$1
  [ -n "$pat" ] || return 0
  while pgrep -f "$pat" > /dev/null; do
    have_time || return 1
    echo "### waiting for $pat ($(date +%H:%M))"
    sleep 300
  done
}

FAILED=()

# ------------------------------------------------------------------------ A-A4
wait_queue "$WAIT_FWD" || { echo "### out of time waiting for rc5f"; exit 1; }

# recipient|donor run ("-" = none)|donor head file ("-" = none)|methods
ALIGN=(
  "f0_local_how2sign_full_k4_s0|f0_local_csl_daily_full_k4_s0|-|procrustes,ridge,learned"
  "f0_local_how2sign_full_k4_s0|f0_local_how2sign+csl_daily-lm_head_full_k4_s0|-|procrustes,ridge,learned"
  "f0_local_how2sign_full_k4_s0|-|mt5_base_pass|learned"
  "f0_local_csl_daily_full_k4_s0|f0_local_how2sign_full_k4_s0|-|procrustes,ridge,learned"
)
for spec in "${ALIGN[@]}"; do
  IFS='|' read -r rcp donor dhead methods <<< "$spec"
  arg=(); tagname=""
  if [ "$donor" != "-" ]; then
    arg=(--donor "$donor"); tagname="$donor"
  else
    arg=(--donor-head "$DONORS/$dhead.run.pth"); tagname="$dhead"
  fi
  # Skip only if every method's donor file is already on disk.
  need=0
  for m in ${methods//,/ }; do
    [ -f "$DONORS/align_${m}_${rcp}__${tagname}.run.pth" ] || need=1
  done
  if [ "$need" -eq 0 ]; then echo "--- align $rcp <- $tagname already built"; else
    wait_gpu || { echo "### out of time before align $rcp"; break; }
    echo "--- align $rcp <- $tagname [$methods]  ($(date +%H:%M))"
    timeout -k 60 "$(secs_to_hard)" "$PY" scripts/29_interface_align.py \
      --recipient "$rcp" "${arg[@]}" --methods "$methods" \
      2>&1 | grep -v "^Loading weights"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { FAILED+=("align $rcp <- $tagname"); continue; }
  fi
  # Score every map that got built, through the same graft path as the paper's
  # own post-hoc rows, so the numbers are directly comparable.
  for m in ${methods//,/ }; do
    f="$DONORS/align_${m}_${rcp}__${tagname}.run.pth"
    [ -f "$f" ] || continue
    out="eval_synthead_align-${m}-${tagname}.json"
    [ -f "runs/$rcp/$out" ] && continue
    wait_gpu || break
    echo "--- score $rcp <- align_$m ($(date +%H:%M))"
    timeout -k 60 "$(secs_to_hard)" "$PY" scripts/05_eval.py \
      --ckpt "runs/$rcp/best.pt" --data data/lex --batch 4 --workers 2 \
      --conditions "$POST_CONDS" --posthoc-head "$f" \
      --out "runs/$rcp/$out" 2>&1 | grep -v "^Loading weights"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { rm -f "runs/$rcp/$out"; FAILED+=("score $out"); }
  done
done

# ------------------------------------------------------------------------ O-P4
wait_queue "$WAIT_TRAIN" || { echo "### out of time waiting for rc5"; exit 1; }

for init in how2sign csl_daily; do
  wait_gpu "$GPU_GRAD_MIB" || { echo "### out of time before grad probe $init"; break; }
  echo "--- grad probe $init ($(date +%H:%M))"
  timeout -k 60 "$(secs_to_hard)" "$PY" scripts/30_grad_probe.py \
    --init "$init" --levels own,mt5_base,csl_daily,how2sign,rand_head_nm \
    2>&1 | grep -v "^Loading weights"
  [ "${PIPESTATUS[0]}" -eq 0 ] || FAILED+=("grad probe $init")
done

echo
[ ${#FAILED[@]} -ne 0 ] && { echo "### failed:"; printf '  %s\n' "${FAILED[@]}"; exit 1; }
echo "### rc5g complete ($(date '+%F %H:%M'))"
