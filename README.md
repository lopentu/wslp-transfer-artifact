# What Actually Transfers? — artifact

Code, item metadata and per-run measurements for

> Kai-Yang Peng, Chunki Lim and Shu-Kai Hsieh. *What Actually Transfers?
> Diagnosing Output-Projection Initialization in Low-Resource Sign Language
> Translation.* WSLP 2026.

The paper rebuilds models from parts of released Uni-Sign checkpoints and asks
which part carries a transfer result. This repository is the measurement trail
behind every number it reports.

## Reproducing the paper's numbers

Every figure in the manuscript is a LaTeX macro generated from the files here:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cd writings/cam-first
PYTHONPATH=../../src ../../.venv/bin/python make_numbers.py --out numbers.tex
```

`make_numbers.py` reads `runs/*/eval_*.json` and `data/`, then writes one
`\newcommand` per reported quantity. `main.tex` cites those macros, so a
disagreement between the paper and these files surfaces as a changed number
rather than as prose no reader can check.

Regenerating `numbers.tex` from this repository alone reproduces 5,230 of the
manuscript's 5,242 macros byte-identically. None of the twelve differences is
cited by the paper; the last section says what they are.

## What is here

| path | what it is |
| --- | --- |
| `src/tsl/` | the library: model assembly, checkpoint surgery, item construction, conditions, metrics |
| `scripts/` | the numbered pipeline, from corpus audit through training, evaluation and analysis |
| `run_*.sh` | the sweeps, each with its cell list and the reasoning for it in the header |
| `runs/<cell>/` | per-item outcomes for every trained cell: forced-choice hits, candidate scores, wrong-clip conditions, training logs |
| `data/` | fold assignment, bootstrap resamples, split statistics, checkpoint geometry, gradient probe, item metadata |
| `writings/cam-first/` | the manuscript source and the generator that fills it |

Cell directories are named `f<fold>_local_<initialization>_full_k4_s<seed>`, so
`f0_local_how2sign+mt5_base-lm_head_full_k4_s0` is fold 0 of How2Sign with
mT5-base's output projection installed.

## What is withheld

Four things are deliberately absent, and each is absent for a reason that also
constrains what could be shipped in its place.

**The Taiwan Sign Language Corpus.** The corpus is a research-use beta release,
and the paper undertakes to redistribute no corpus material or signer metadata.
No utterance, translation, context sentence or signer label appears anywhere in
this repository. What the item records keep is everything the analysis reads and
nothing that reads back as language:

- gold words and distractor pools become stable ids (`w3a91f0c2e1`), so item
  counts, type counts and pool sizes stay exact while the words are gone;
- pointing-sign glosses become person labels (`1SG`, `2PL`, `3`), which is the
  distinction the analysis actually tests;
- clip boundaries are rebased to zero, so durations — which drive the
  duration-matched wrong-clip pairing — are exact, while absolute offsets into
  the recordings are not released;
- utterance text, context, signer and theme are dropped outright.

`tools/leakscan.py` is how that was enforced, and how it can be audited. It
compares every file in a tree against the corpus's full utterance inventory,
and packaging runs it against the unredacted corpus as its last step. It ships
so the undertaking can be checked rather than taken on trust: anyone with a
corpus licence can point it at this repository and reach the same verdict.

```bash
python tools/leakscan.py --records /path/to/unredacted/records.jsonl \
    'data/*.json' 'runs/*/*.json' 'src/tsl/*.py'
```

Pointed at the item file released here it refuses to run rather than compare
redacted ids against redacted ids and report a reassuring nothing. For holders
of a corpus licence, `scripts/16_build_lexitems.py` and
`scripts/17_lm_match_distractors.py` rebuild the full item set.

**Donor checkpoints.** The released Uni-Sign checkpoints this work recombines
carry no redistribution licence, and neither do weights derived from them.
Nothing here is a parameter: the longest numeric array in `runs/` is the four
candidate log-probabilities of a single forced choice, and `data/` holds
statistics about the checkpoints — drift, cosine, conditioning — never their
contents. `scripts/09_fetch_unisign.py` fetches the upstream source, and the
commit it was pinned to is recorded in `third_party/unisign/README.NOTICE`.

**Free-decoding output.** `eval_gen.json` pairs each hypothesis with its corpus
reference, so it cannot ship — and BLEU is not recomputable without references
under any redaction. The statistics it fed are therefore precomputed per cell in
`data/gen_metrics.json`: BLEU, chrF, distinct-output share, modal-output share
and the two Han-character shares.

**ASL-LEX.** The iconicity ratings are a third-party database with its own
terms. Place `signdata.csv` at `data/external/asllex/` to resolve the three
population-level iconicity macros.

## What the redaction costs

Withholding those inputs has exactly two visible consequences.

First, three places in `make_numbers.py` had to change, each marked `# RELEASE:`
in the file: the person-label substitution, so the script's internal consistency
assertion tests the same equivalence against the public records; the fallback to
`data/gen_metrics.json` for free decoding; and a guard so a missing ASL-LEX
resolves to a placeholder instead of raising. Nothing else in `src/`, `scripts/`
or the sweeps differs, apart from two docstrings that had quoted a corpus
sentence to illustrate a failure mode — the mechanism each describes survives in
words.

Second, twelve macros no longer regenerate, none of them cited by the paper.
Three are the ASL-LEX population statistics. Four belong to a multiplicity
correction whose family `make_numbers.py` reads from an earlier draft that is
not part of this release. The remaining five are provenance strings resolved by
scripts that need the upstream source.

## Licence

Code — `src/`, `scripts/`, the sweeps and `make_numbers.py` — is released under
the MIT Licence (`LICENSE`). The measurements in `runs/`, the derived analysis
files in `data/` and the redacted item metadata are released under CC BY 4.0
(`LICENSE-DATA`): reuse them freely, and cite the paper.

The Taiwan Sign Language Corpus, the Uni-Sign checkpoints and ASL-LEX are
governed by their own terms and are covered by neither.
