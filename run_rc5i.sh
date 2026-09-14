#!/usr/bin/env bash
# Everything left after the training queue and the forward queue: the two
# mechanism probes, the drift-matched written-language instrument, and a
# re-entry into both earlier queues for the cells that were gated on a donor
# another queue had to build first.
#
# Replaces run_rc5g.sh and run_rc5h.sh, which were killed at 22:14 for a reason
# worth recording. They sequenced themselves with `pgrep -f '[r]un_rc5[.]running'`
# -- the bracket trick, correctly applied to the script's own name. But the
# pattern also matched the *wrapper shell* that launched the queue, whose command
# line contains the snapshot's filename and which outlives the queue it started.
# So the wait would never have cleared and both queues would have sat there until
# the deadline. The bracket trick stops a pattern matching its own pgrep; it does
# nothing about a pattern matching whatever else happens to quote the name.
#
# So this script gates on the PYTHON PROCESS instead of the script name. There is
# exactly one thing a training cell can be, `scripts/04_train.py`, and no wrapper
# quotes it. That is also the right gate on the merits: what the gradient probe
# must not collide with is a training cell, not a particular queue.
#
#   cp run_rc5i.sh .run_rc5i.running.sh && nohup ./.run_rc5i.running.sh \
#       > runs/logs/rc5_probe.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PLAN_BY=${PLAN_BY:-2026-08-24 12:00}
HARD_STOP=${HARD_STOP:-2026-08-24 14:00}
GPU_EVAL_MIB=${GPU_EVAL_MIB:-7500}
GPU_GRAD_MIB=${GPU_GRAD_MIB:-13000}
GPU_LM_MIB=${GPU_LM_MIB:-11000}
GPU_TRAIN_MIB=${GPU_TRAIN_MIB:-16000}
HOLD_FILE=${HOLD_FILE:-.gpu-hold}
POST_CONDS=${POST_CONDS:-local,swap_plain@0,swap_plain@1,swap_plain@2}
DONORS=data/external/head_donors
DRIFT_TARGET=${DRIFT_TARGET:-0.85}

PLAN_TS=$(date -d "$PLAN_BY" +%s) || exit 1
HARD_TS=$(date -d "$HARD_STOP" +%s) || exit 1
secs_to_hard() { echo $(( HARD_TS - $(date +%s) )); }
have_time() { [ "$(( PLAN_TS - $(date +%s) ))" -ge 900 ]; }
gpu_free()  { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

wait_gpu() {
  local need=${1:-$GPU_EVAL_MIB} waited=0
  while :; do
    [ -f "$HOLD_FILE" ] || { [ "$(gpu_free)" -ge "$need" ] && return 0; }
    have_time || return 1
    [ $(( waited % 1800 )) -eq 0 ] && echo "### waiting for ${need} MiB ($(date '+%m-%d %H:%M'))"
    sleep 60; waited=$(( waited + 60 ))
  done
}

# Gate on the python process, never on a script name: see the header.
busy() { pgrep -f "[0]4_train[.]py" > /dev/null || pgrep -f "[2]8_gloss_lm[.]py" > /dev/null; }
wait_idle() {
  local waited=0
  while busy; do
    have_time || return 1
    [ $(( waited % 1800 )) -eq 0 ] && echo "### waiting for the training queue to idle ($(date '+%m-%d %H:%M'))"
    sleep 120; waited=$(( waited + 120 ))
  done
}

FAILED=()

# ------------------------------------------------------- stage 1: A-A4, 7.5 GB
# Holds one model at a time, so it can run beside a training cell (14.4 GB peak
# + 7.5 = 21.9 of 24.5).
ALIGN=(
  "f0_local_how2sign_full_k4_s0|f0_local_csl_daily_full_k4_s0|-|procrustes,ridge,learned"
  "f0_local_how2sign_full_k4_s0|f0_local_how2sign+csl_daily-lm_head_full_k4_s0|-|procrustes,ridge,learned"
  "f0_local_how2sign_full_k4_s0|-|mt5_base_pass|learned"
  "f0_local_csl_daily_full_k4_s0|f0_local_how2sign_full_k4_s0|-|procrustes,ridge,learned"
)
for spec in "${ALIGN[@]}"; do
  IFS='|' read -r rcp donor dhead methods <<< "$spec"
  if [ "$donor" != "-" ]; then arg=(--donor "$donor"); tagname="$donor"
  else arg=(--donor-head "$DONORS/$dhead.run.pth"); tagname="$dhead"; fi
  need=0
  for m in ${methods//,/ }; do
    [ -f "$DONORS/align_${m}_${rcp}__${tagname}.run.pth" ] || need=1
  done
  if [ "$need" -eq 0 ]; then echo "--- align $rcp <- $tagname already built"; else
    wait_gpu "$GPU_EVAL_MIB" || { echo "### out of time before align"; break; }
    echo "--- align $rcp <- $tagname [$methods]  ($(date '+%m-%d %H:%M'))"
    timeout -k 60 "$(secs_to_hard)" "$PY" scripts/29_interface_align.py \
      --recipient "$rcp" "${arg[@]}" --methods "$methods" \
      2>&1 | grep -v "^Loading weights"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { FAILED+=("align $rcp <- $tagname"); continue; }
  fi
  for m in ${methods//,/ }; do
    f="$DONORS/align_${m}_${rcp}__${tagname}.run.pth"
    [ -f "$f" ] || continue
    out="eval_synthead_align-${m}-${tagname}.json"
    [ -f "runs/$rcp/$out" ] && continue
    wait_gpu "$GPU_EVAL_MIB" || break
    echo "--- score $rcp <- align_$m ($(date '+%m-%d %H:%M'))"
    timeout -k 60 "$(secs_to_hard)" "$PY" scripts/05_eval.py \
      --ckpt "runs/$rcp/best.pt" --data data/lex --batch 4 --workers 2 \
      --conditions "$POST_CONDS" --posthoc-head "$f" \
      --out "runs/$rcp/$out" 2>&1 | grep -v "^Loading weights"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { rm -f "runs/$rcp/$out"; FAILED+=("score $out"); }
  done
done

# ---------------------------- gate: the earlier queues must be FINISHED
# Not merely idle. `cell_done` stops a queue redoing a finished cell, but two
# LIVE queues would both see an unfinished cell as unclaimed and train it
# twice; and between cells a queue sits in `wait_gpu` with no python child, so
# a process-name gate reads as idle. Everything below this line either trains
# (16 GB) or fine-tunes a full mT5 (11 GB), and either would race a 14.4 GB
# training cell into an OOM that takes both jobs down -- the two `wait_gpu`
# polls are 60 s apart and can both see the same free memory.
#
# Gate on the queues' PIDs, passed in at launch. A PID cannot be matched by a
# wrapper shell that merely quotes a filename, which is the trap that cost
# run_rc5g.sh and run_rc5h.sh their entire run.
#
# 8/23 03:20 -- and pass ONLY the queue's own pid, never the launcher's. This
# script was first started with WAIT_PIDS="1171872 1171877 1227513 1227517" on
# the reasoning that a waiter over extra pids can only wait longer. That is false
# when one of them is the queue's PARENT: 1227513 is Claude Code's wrapper shell,
# it was orphaned to init with its command long finished, and after five hours it
# had not exited and was never going to. So stages 2 and 3 -- the drift-matched
# instrument for Reviewer G and the O-P4 gradient probe -- were never going to
# run, and the log looked exactly like patient waiting.
#
# Resolve the queue, and nothing else, with
#   ps -eo pid,args | awk '$2=="bash" && $3=="./.run_rc5x.running.sh" {print $1}'
wait_pids() {
  local pid waited=0
  for pid in $WAIT_PIDS; do
    while kill -0 "$pid" 2>/dev/null; do
      have_time || return 1
      [ $(( waited % 1800 )) -eq 0 ] && echo "### waiting for queue pid $pid ($(date '+%m-%d %H:%M'))"
      sleep 120; waited=$(( waited + 120 ))
    done
    echo "### queue pid $pid has exited"
  done
}

if [ -n "${WAIT_PIDS:-}" ]; then
  wait_pids || { echo "### out of time waiting for the earlier queues"; }
fi

# ------------------------------------ stage 2: G / A-B2 drift-matched instrument
# Needs a full mT5 with AdamW, so it waits for a gap rather than sharing.
STRONG=data/external/tsl_text_lm_strong.pth
if [ ! -f "$STRONG" ]; then
  for rung in "3e-4|24" "1e-3|24" "3e-3|24"; do
    IFS='|' read -r lr ep <<< "$rung"
    rep="data/text_lm_strong_lr${lr}.json"
    if [ ! -f "$rep" ]; then
      wait_idle && wait_gpu "$GPU_LM_MIB" || break
      echo "--- tsl_text_strong lr=$lr epochs=$ep ($(date '+%m-%d %H:%M'))"
      timeout -k 60 "$(secs_to_hard)" "$PY" scripts/23_text_lm.py \
        --fold 0 --lr "$lr" --epochs "$ep" \
        --out "data/external/tsl_text_lm_strong_lr${lr}.pth" --report "$rep" \
        2>&1 | grep -v "^Loading weights"
      [ -f "$rep" ] || { echo "### rung $lr failed"; FAILED+=("text_lm lr=$lr"); continue; }
    fi
    drift=$("$PY" -c "import json;print(json.load(open('$rep'))['lm_head_rel_l2_from_mt5base'])")
    echo "### lr=$lr drift=$drift (target >= $DRIFT_TARGET)"
    if [ "$("$PY" -c "print(1 if float('$drift')>=float('$DRIFT_TARGET') else 0)")" = "1" ]; then
      cp "data/external/tsl_text_lm_strong_lr${lr}.pth" "$STRONG"
      echo "### drift-matched at lr=$lr ($drift) -> $STRONG"; break
    fi
  done
  # No rung cleared the target: take the furthest and let the write-up state the
  # drift it reached. A partly-matched instrument with its number printed is
  # worth more than a gap.
  if [ ! -f "$STRONG" ]; then
    lr=$("$PY" - <<'EOF'
import glob, json, re
rows = []
for p in glob.glob("data/text_lm_strong_lr*.json"):
    try: rows.append((json.load(open(p))["lm_head_rel_l2_from_mt5base"], p))
    except Exception: pass
print(re.sub(r".*_lr(.*)\.json", r"\1", max(rows)[1]) if rows else "")
EOF
)
    if [ -n "$lr" ] && [ -f "data/external/tsl_text_lm_strong_lr${lr}.pth" ]; then
      cp "data/external/tsl_text_lm_strong_lr${lr}.pth" "$STRONG"
      echo "### no rung reached $DRIFT_TARGET; using the furthest (lr=$lr)"
    fi
  fi
fi

if [ -f "$STRONG" ]; then
  name=f0_local_how2sign+tsl_text_strong-lm_head_full_k4_s0
  if [ ! -f "runs/$name/best.pt" ]; then
    if wait_idle && wait_gpu "$GPU_TRAIN_MIB"; then
      echo "--- $name ($(date '+%m-%d %H:%M'))"
      timeout -k 60 "$(secs_to_hard)" "$PY" scripts/04_train.py \
        --fold 0 --system local --init how2sign --init-parts all \
        --init-mt5 tsl_text_strong --init-mt5-parts lm_head \
        --tune full --ctx-k 4 --epochs 12 --seed 0 2>&1 | grep -v "^Loading weights"
    fi
  fi
  for spec in "eval.json|data|local,text_only" \
              "eval_lex.json|data/lex|local,blank_plain" \
              "eval_clip.json|data/lex|local,blank_plain,swap_plain@0,swap_plain@1,swap_plain@2,shuffle_frames"; do
    IFS='|' read -r out data conds <<< "$spec"
    [ -f "runs/$name/$out" ] && continue
    [ -f "runs/$name/best.pt" ] || continue
    wait_gpu "$GPU_EVAL_MIB" || break
    timeout -k 60 "$(secs_to_hard)" "$PY" scripts/05_eval.py \
      --ckpt "runs/$name/best.pt" --data "$data" --batch 4 --workers 2 \
      --conditions "$conds" --out "runs/$name/$out" 2>&1 | grep -v "^Loading weights"
  done
fi

# ----------------------------------------------- stage 3: re-entry, then O-P4
# Two queues must be FINISHED, not merely idle, before this re-enters them:
# `cell_done` stops a queue redoing a finished cell, but two live queues would
# both see an unfinished cell as unclaimed and train it twice. Idleness is not
# enough either — between cells a queue sits in `wait_gpu` with no python child.
#
# So gate on the queues' PIDs, passed in at launch. A PID is the one identifier
# that cannot be matched by a wrapper shell that merely quotes a filename, which
# is the trap that cost run_rc5g.sh and run_rc5h.sh their whole run.
echo "### re-entering the training queue for the donor-gated cells ($(date '+%m-%d %H:%M'))"
cp run_rc5.sh .run_rc5re.running.sh && chmod +x .run_rc5re.running.sh
PLAN_BY="$PLAN_BY" HARD_STOP="$HARD_STOP" ./.run_rc5re.running.sh 2>&1 | grep -vE "^Loading weights" || true

echo "### re-entering the forward queue for the remaining decodes ($(date '+%m-%d %H:%M'))"
cp run_rc5f.sh .run_rc5fre.running.sh && chmod +x .run_rc5fre.running.sh
PLAN_BY="$PLAN_BY" HARD_STOP="$HARD_STOP" ./.run_rc5fre.running.sh 2>&1 | grep -vE "^Loading weights" || true

# The gradient probe last: it backpropagates through the full 582M mT5, so it
# cannot share the card with a training cell, and nothing else is waiting on it.
for init in how2sign csl_daily; do
  wait_idle || break
  wait_gpu "$GPU_GRAD_MIB" || break
  echo "--- grad probe $init ($(date '+%m-%d %H:%M'))"
  timeout -k 60 "$(secs_to_hard)" "$PY" scripts/30_grad_probe.py \
    --init "$init" --levels own,mt5_base,csl_daily,how2sign,rand_head_nm \
    2>&1 | grep -v "^Loading weights"
  [ "${PIPESTATUS[0]}" -eq 0 ] || FAILED+=("grad probe $init")
done

echo
[ ${#FAILED[@]} -ne 0 ] && { echo "### failed:"; printf '  %s\n' "${FAILED[@]}"; exit 1; }
echo "### rc5i complete ($(date '+%F %H:%M'))"
