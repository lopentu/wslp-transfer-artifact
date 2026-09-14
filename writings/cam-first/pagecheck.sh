#!/usr/bin/env bash
# Page-budget probe, camera-ready edition.
#
# WSLP's camera-ready allowance for a short paper: 5 pages of content (4 + the
# one extra page granted for addressing reviewers), with Broader Impacts/Ethics
# and Limitations optionally spilling onto a 6th, and references after that with
# no limit. So there are two separate ceilings, and the old submission-time probe
# checked neither of them:
#
#   Limitations must OPEN on page 5 or earlier   -> the body fits in 5 pages
#   References  must OPEN on page 6 or earlier   -> Limitations+Ethics fit in 1
#
# The old probe asked whether page 5's FIRST LINE is "References", which is a
# false negative whenever References opens near the bottom of the preceding page
# -- exactly where it opens in this paper. Locate the headings instead, and
# report the fraction of the way down the page each one falls, because "fits" and
# "has room left" are different questions and only the second one tells us
# whether another figure can go in.
#
# Unresolved macros still render as `[TODO: ...]`, ten times the width of the
# number that will replace them, so the probe substitutes a number-width
# placeholder first. It never touches numbers.tex or build/.
set -euo pipefail
cd "$(dirname "$0")"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/out"
# Every .tex the document \input{}s, not just main.tex: a missing figure file
# makes tectonic fail inside the temp dir, and the probe would then report a
# page-budget failure for what is actually a copy that was never made.
cp main.tex fig-protocol.tex refs.bib acl.sty acl_natbib.bst "$T/"
sed 's/\\TODO{[^}]*}/00.0/g' numbers.tex > "$T/numbers.tex"
./tectonic -X compile "$T/main.tex" --outdir "$T/out" > /dev/null 2>&1
../../.venv/bin/python - "$T/out/main.pdf" <<'PY'
import sys, pypdf

BODY_MAX, EXTRA_MAX = 5, 6
r = pypdf.PdfReader(sys.argv[1])
pages = [(p.extract_text() or "").split("\n") for p in r.pages]


def find(heading):
    """(1-indexed page, fraction down, index among non-empty lines) of a heading."""
    for i, lines in enumerate(pages):
        seen = 0
        for j, line in enumerate(lines):
            if not line.strip():
                continue
            seen += 1
            if line.strip() == heading:
                return i + 1, (j + 1) / max(len(lines), 1), seen
    return None, None, None


print(f"pages: {len(r.pages)}")
for i, lines in enumerate(pages[:8]):
    first = next((l for l in lines if l.strip()), "")
    print(f"  p{i+1}: {first[:56]}")

ok = True
print()

# The body budget is "content fits in BODY_MAX pages", and Limitations is the
# first thing that is NOT content. So the body fits when Limitations opens on
# page BODY_MAX or earlier, and ALSO when it opens at the very top of the next
# page -- that case means the body ended exactly at the bottom of page BODY_MAX,
# which is the budget spent precisely, not exceeded. Testing `page <= BODY_MAX`
# alone rejects it, and that is a false negative of the same family as the one
# this probe was rewritten to remove.
lim_pg, lim_frac, lim_rank = find("Limitations")
if lim_pg is None:
    print("  Limitations  NOT FOUND — cannot verify the body budget")
    ok = False
else:
    # "At the top" must mean the heading is literally the first thing typeset on
    # the page -- not merely high on it. A fractional tolerance let nine lines of
    # Discussion sit above the heading and still pass, which is the overrun this
    # check exists to catch, so test the heading's rank among non-empty lines.
    at_top = lim_pg == BODY_MAX + 1 and lim_rank == 1
    body_ok = lim_pg <= BODY_MAX or at_top
    note = "  (body ends exactly at the p%d boundary)" % BODY_MAX if at_top else ""
    print(f"  {'Limitations':<12} opens p{lim_pg} ({lim_frac:.0%} down)  "
          f"body limit p{BODY_MAX}  [{'ok' if body_ok else 'OVER'}]{note}")
    ok &= body_ok

ref_pg0, ref_frac0, _ = find("References")
if ref_pg0 is None:
    print("  References   NOT FOUND — cannot verify the overall budget")
    ok = False
else:
    print(f"  {'References':<12} opens p{ref_pg0} ({ref_frac0:.0%} down)  "
          f"limit p{EXTRA_MAX}  [{'ok' if ref_pg0 <= EXTRA_MAX else 'OVER'}]")
    ok &= ref_pg0 <= EXTRA_MAX

# How much of the allowance is actually spent: the last page before References
# that still carries content, against the 6-page ceiling.
ref_pg, ref_frac, _ = find("References")
if ref_pg:
    print(f"\n  content ends {ref_frac:.0%} down p{ref_pg}; "
          f"{'room for roughly %.1f more column-pages' % ((EXTRA_MAX - ref_pg) + (1 - ref_frac))
             if ref_pg <= EXTRA_MAX else 'OVER BUDGET'}")

print("\nFITS" if ok else "\nOVER BUDGET")
sys.exit(0 if ok else 1)
PY
