"""Human iconicity ratings for the concepts our lexical items are built on.

The lexical diagnostic (`lexitems.py`) replaces pointing signs with content
words, which removes the *indexical* confound. It does not remove the **iconic**
one: unrelated sign languages converge independently on signs motivated by
resemblance — TSL and ASL both mime a steering wheel for 開車 — so a lexical set
made mostly of iconic signs would reproduce the pointing problem one level down.
Iconicity therefore has to be a reported stratum.

## Why this module exists rather than the SignCLIP one it replaces

The first attempt derived the stratum from the SignCLIP artefacts the JSL probe
produced: cosine between a TSL clip and the same concept's text embedding in each
of 41 sign languages, z-scored within the gloss. It looked convincing on the
tails — 開車, 魚, 笑 at the top, 高中, 中秋節, 國小 at the bottom — and it failed
when tested against this file. Measured 2026-08-14:

    corr(SignCLIP asl_z, ASL-LEX human iconicity)
        pearson +0.100   spearman +0.087   n = 1,189

One percent of the variance. The tercile means are monotone (+0.278 / +0.334 /
+0.420), so it is not pure noise, but nothing that weak can carry a
stratification, and the tails that made it look right were a hand-picked dozen
out of a distribution that is otherwise unrelated to the construct.

A second problem is structural and not fixable by more data: **per-gloss
reliability cannot be computed at all**. Only 4 of twtsl's 3,428 glosses have two
or more clips, so every gloss's score rests on a single video of a single signer.
The probe's split-half figure of 0.45 is for the *language-level* ranking over
thousands of clips and says nothing about whether one entry's score is stable.

So the SignCLIP fields are still written onto each record, because a measurement
that has been made should stay inspectable, but they are secondary and must not
be used to stratify or reported as form similarity.

## What this measure is, stated precisely

`Iconicity(M)` is how iconic raters judged the **ASL** sign for a concept to be.
It is not a measurement of how similar the TSL and ASL signs are, and the paper
must not describe it as one. It is a proxy for *which concepts invite unrelated
sign languages to converge*, which is the confounding mechanism — and unlike the
embedding it is a published, human-rated, citable instrument with its own SD and
rater counts.

The honest residual: the proxy is one step removed from the quantity of interest,
and mapping Chinese glosses onto English headwords onto ASL-LEX entries adds
translation noise at two joins. The direct instrument would compare twtsl's coded
location and handshape against ASL-LEX's `MajorLocation.2.0` and `Handshape.2.0`
— two independently human-coded lexicons, and a genuine form comparison. That
needs the two coding schemes reconciled, which is more work than the remaining
days allow.

## The join, which has bitten this project once already

twtsl's `entries` table is keyed `(lang, id, row_id)`, so `id` is **scoped per
language** and joining Chinese to English on it silently produces garbage: the
pairing (id, row_id) yields twelve rows, of which 幾天 ↔ A FUNERAL PROCESSION and
算盤 ↔ SHOT-PUT are representative. Both rows of a real pair point at the same
`clip`, and that is the only correct key — the same conclusion
`13_signclip_poses.py` reached after the same mistake.
"""

from __future__ import annotations

import csv
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from .strata import TWTSL_DB, normalise_gloss

ASLLEX_CSV = Path("data/external/asllex/signdata.csv")

# ASL-LEX ships as latin-1: a handful of entries carry Windows-1252 punctuation
# and utf-8 decoding dies on them mid-file.
ASLLEX_ENCODING = "latin-1"


def _key(s: str | None) -> str:
    """Normalise an English headword for matching across the two lexicons."""
    if not s:
        return ""
    return re.sub(r"[^a-z ]", "", str(s).lower().replace("_", " ")).strip()


def zh_to_en(db_path: Path | str = TWTSL_DB) -> dict[str, list[str]]:
    """Chinese gloss -> English headwords, joined on `clip`.

    Never on `id`; see the module docstring. Variant suffixes are stripped from
    both sides with `normalise_gloss`, so 會_N and 會_S both key 會 and both
    contribute their English headword.
    """
    con = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    by_clip: dict[str, dict[str, str]] = defaultdict(dict)
    for lang, name, clip in con.execute(
            "SELECT lang, name, clip FROM entries WHERE clip IS NOT NULL"):
        if lang in ("zh", "en") and name:
            by_clip[clip][lang] = name
    out: dict[str, list[str]] = defaultdict(list)
    for pair in by_clip.values():
        if "zh" in pair and "en" in pair:
            z = normalise_gloss(pair["zh"])
            e = _key(normalise_gloss(pair["en"]))
            if z and e and e not in out[z]:
                out[z].append(e)
    return dict(out)


def ratings(csv_path: Path | str = ASLLEX_CSV) -> dict[str, dict]:
    """English headword -> mean iconicity, its SD and rater count.

    Two tiers, and the distinction matters. `LemmaID`/`EntryID` name the sign
    itself; `DominantTranslation` is what raters most often *called* it, and
    several distinct signs can share one. Pooling all three averages unrelated
    signs together: "think" is the lemma THINK (6.58) but also the dominant
    translation of OFFHAND (1.92) and PONDER (4.10), and the flat mean of 4.80
    describes none of them. So lemma matches win outright, and a translation
    match is used only where no lemma matches — recorded in `source` so the
    weaker tier can be excluded or reported separately.

    Where several entries still share a key they are averaged, with `n_entries`
    recording how many, so an ambiguous concept stays visible.
    """
    p = Path(csv_path)
    if not p.exists():
        return {}
    tiers: dict[str, dict[str, list[tuple[float, float, float]]]] = {
        "lemma": defaultdict(list), "translation": defaultdict(list)}
    with p.open(newline="", encoding=ASLLEX_ENCODING) as fh:
        for row in csv.DictReader(fh):
            try:
                m = float(row["Iconicity(M)"])
            except (KeyError, TypeError, ValueError):
                continue

            def _f(name: str) -> float:
                try:
                    return float(row.get(name) or "nan")
                except ValueError:
                    return float("nan")

            rec = (m, _f("Iconicity(SD)"), _f("Iconicity(N)"))
            for field, tier in (("LemmaID", "lemma"), ("EntryID", "lemma"),
                                ("DominantTranslation", "translation")):
                k = _key(row.get(field))
                if k:
                    tiers[tier][k].append(rec)
    out: dict[str, dict] = {}
    for tier in ("translation", "lemma"):   # lemma written last, so it wins
        for k, v in tiers[tier].items():
            out[k] = {
                "icon": sum(x[0] for x in v) / len(v),
                "icon_sd": v[0][1],
                "icon_n": v[0][2],
                "n_entries": len(v),
                "source": tier,
            }
    return out


def iconicity_by_zh(db_path: Path | str = TWTSL_DB,
                    csv_path: Path | str = ASLLEX_CSV) -> dict[str, dict]:
    """Chinese gloss -> iconicity of the ASL sign for the same concept.

    A Chinese gloss reaching several English headwords (free variants of one
    lexeme) is averaged over whichever of them ASL-LEX rates, and `n_en` keeps
    the ambiguity on the record.
    """
    rate = ratings(csv_path)
    if not rate:
        return {}
    out = {}
    for z, ens in zh_to_en(db_path).items():
        hits = [rate[e] for e in ens if e in rate]
        if not hits:
            continue
        out[z] = {
            "icon": sum(h["icon"] for h in hits) / len(hits),
            "icon_n": hits[0]["icon_n"],
            "n_en": len(ens),
            "n_rated": len(hits),
            "en": next(e for e in ens if e in rate),
            "source": "lemma" if any(h["source"] == "lemma" for h in hits)
                      else "translation",
        }
    return out
