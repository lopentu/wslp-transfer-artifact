#!/usr/bin/env bash
# A second text-only LM on the finished lexical item set.
#
# The rank balance is a property of (item set x scoring rule x scoring model).
# The set was built so that Qwen2.5-1.5B-Instruct -- the model that chose the
# distractors -- is at 25.0% by construction, and Appendix A already says that the
# guarantee is specific to the scoring rule. What it cannot say from one model is
# whether the balance generalises: a reader is entitled to suspect that "a
# text-only LLM is at chance" means "the one LLM we balanced against is at chance".
#
# So: a different family, not a different checkpoint of the same one. GLM-Edge-1.5B
# (Zhipu) against Qwen2.5-1.5B-Instruct (Alibaba) -- same scale, unrelated
# tokenizer and pretraining corpus, both natively Chinese. Distractors are NOT
# re-chosen; this scores the finished set exactly as published.
#
#   LM-B near 25%     the balance transfers, and the claim can stay as it is.
#   LM-B at 30-40%    the claim narrows to "balanced against the designated
#                     construction model", with this figure reported beside it.
#
# fp32 on CPU, matching how the published 25.0% was measured (a bf16/GPU pass of
# the same set gave 25.7 -- precision, not disagreement). It therefore needs ~7 GB
# of host RAM and no card at all, which is why it waits for our own GPU jobs to
# release their dataloader workers first: this box has 31 GB and 15 logged-in
# users, and an OOM kill would take the sweep with it.
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
LLM=${LLM:-zai-org/glm-edge-1.5b-chat}
OUT=${OUT:-data/lex/shortcuts.glm.json}
NOT_BEFORE=${NOT_BEFORE:-20:50}
NEED_GB=${NEED_GB:-9}

target=$(date -d "today $NOT_BEFORE" +%s)
[ "$target" -le "$(date +%s)" ] || {
  echo "### waiting until $NOT_BEFORE for the GPU sweeps to end  ($(date +%H:%M))"
  sleep $(( target - $(date +%s) ))
}
while pgrep -f "scripts/0[45]_(train|eval).py" > /dev/null; do
  echo "### a train/eval process is still up — waiting  ($(date +%H:%M))"; sleep 300
done
while :; do
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  [ "${avail:-0}" -ge "$NEED_GB" ] && break
  echo "### only ${avail} GB available, need ${NEED_GB}  ($(date +%H:%M))"; sleep 300
done

echo "### second-LM check: $LLM on data/lex, fp32/CPU  ($(date +%H:%M))"
"$PY" scripts/07_shortcuts.py --data data/lex --ctx-k 4 --device cpu --threads 8 \
  --llm "$LLM" --out "$OUT"
echo "### -> $OUT  ($(date +%H:%M))"
