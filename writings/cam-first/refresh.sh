#!/usr/bin/env bash
# Re-derive everything the camera-ready reads, in dependency order, then rebuild.
#
# Six analyses sit between the run outputs and the macros, and the camera-ready
# sweep lands cells one at a time, so running these in the wrong order (or
# forgetting one) produces a paper whose text and tables disagree about which
# cells exist. The order below is the dependency order.
#
#   ./refresh.sh            # split stats + bootstrap + numbers + build + pages
#   ./refresh.sh --quick    # numbers + build + pages only (no re-analysis)
#
# The bootstrap is the slow step (10,000 paragraph-clustered resamples over every
# registered contrast) and is the one that must be rerun whenever a new cell
# lands, because that is where the new cells' confidence intervals come from.
set -uo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
ROOT=../..

if [ "${1:-}" != "--quick" ]; then
  echo "== split sizes and signer allocation"
  "$PY" "$ROOT/scripts/34_split_stats.py"
  echo
  echo "== inter-model agreement"
  "$PY" "$ROOT/scripts/18_agreement.py" | tail -n 8
  echo
  echo "== paragraph-clustered bootstrap and permutation tests"
  "$PY" "$ROOT/scripts/19_bootstrap.py" | tail -n 12
  echo
fi

echo "== numbers"
"$PY" make_numbers.py --check
rc=$?
echo
echo "== build"
./tectonic -X compile main.tex --outdir build 2>&1 | grep -E "^error|Writing" || true
echo
echo "== page budget"
./pagecheck.sh
page=$?

echo
if [ "$rc" -ne 0 ]; then
  echo "!! the paper cites macros with no run behind them (listed above)."
  echo "   Expected while the sweep is running; a bug at submission."
fi
[ "$page" -ne 0 ] && echo "!! over the camera-ready page budget — see pagecheck.sh."
exit $(( rc != 0 || page != 0 ))
