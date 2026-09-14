"""Difficulty strata — where an encoder-side benefit has to land, and a
decoder-side one does not.

This is axis 3 (README §4), and it is the only test of the genealogy hypothesis
available without a JSL checkpoint. The transfer table on its own cannot separate
two accounts of a CSL advantage:

* **decoder credit** — the CSL checkpoint's mT5 already emits fluent Chinese, so
  it helps wherever fluent Chinese helps, evenly, regardless of the sign form;
* **encoder credit** — its pose encoder has seen sign language, so it helps most
  exactly where the *form* has to be read carefully.

The two make opposite predictions once utterances are split by how hard their
signs are to read visually. So we need a difficulty measure that is computed from
data already in hand, is objective, and is not itself a translation model.

Two are available.

**Phonological neighbourhood density.** twtsl codes every entry with a location
and a handshape per slot; signs sharing a slot-1 (location, handshape) are the
ones that look alike at the coarsest level. A sign in a crowded neighbourhood has
many visual competitors, so reading it demands more of the encoder. Median
neighbourhood is 11 signs and the largest is 181, so this genuinely separates.

**Duration reduction.** A sign produced quickly is coarticulated, and a
coarticulated token is the one an encoder trained on citation-like data has least
to say about. Reduction here is a token's duration over the **median duration of
that same sign type across the corpus** — so it asks "was this production rushed
*for this sign*", and a sign that is simply long by nature does not count as
unreduced.

That is deliberately not the definition README once carried, which was running
duration over *citation* duration taken from the dictionary clips. That version
was built, measured and rejected, and the rejection is reproducible with
`scripts/12_strata.py --audit-citation`:

* Raw clip length is hopeless — clips run a median 3.44 s against a median
  running token of 0.72 s, because the signer stands in frame before and after
  signing. Rank correlation with running duration across signs: **+0.04**.
* Isolating the moving part with a frame-difference envelope helps, but the
  envelope shows why it cannot be enough: the mean clip contains **2.2 motion
  bursts**, because the dictionary convention is to produce each sign twice. A
  first-to-last active span therefore measures two productions plus the gap
  (+0.12); taking a single burst gets to **+0.21** and no threshold does better.
* At that point the measured citation burst (700 ms) is indistinguishable from
  the running median (685 ms), so the ratio has no headroom left to express
  reduction with.

The within-corpus measure is what survives the equivalent check. Its split-half
reliability over utterances is **+0.39, Spearman-Brown +0.56**, it covers 67% of
tokens against the citation join's 45%, and it spans p10 0.72 to p90 1.45. A real
citation baseline would need pose-level segmentation of the dictionary clips with
a hold/movement model, which is not eight days' work.

Nothing here is a label, and neither measure is computed from anything a model
produced, so splitting results by them cannot be circular. Each record also
carries its token count and duration, because "reduced utterances are harder" and
"short utterances are harder" are different claims and the report has to be able
to tell them apart.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

TWTSL_DB = Path("/mnt/md0/corpus/sign/twtsl/twtsl.db")
TWTSL_MEDIA = Path("/mnt/md0/corpus/sign/twtsl/media")

# twtsl marks variants with a trailing single-letter suffix: _A/_B/_C/_D/_E are
# free variants, _N/_S are the northern and southern forms. They are variants of
# one lexeme, and the continuous corpus writes the bare gloss, so a type-level
# lookup has to strip them. What it must NOT do is average over them silently:
# only 23% of multi-variant lexemes share a slot-1 signature — `會_N` is signed at
# the back of the hand and `會_S` at the temple — so a gloss with disagreeing
# variants is genuinely ambiguous as to which form was produced. Every record
# therefore carries how many distinct signatures its tokens could have had, and
# the report can drop the ambiguous ones instead of quietly averaging them.
_VARIANT = re.compile(r"_[A-EGNS]\s*$")
_PAREN = re.compile(r"[（(][^）)]*[）)]")


def normalise_gloss(gloss: str | None) -> str:
    """twtsl `丁(姓)_A` and corpus `(一包)給` -> a common key.

    Parentheses are disambiguators on the twtsl side and classifier notes on the
    corpus side; neither is part of the sign's identity. Stripping them can empty
    a gloss that is *only* a note (`(手勢)`, "gesture"), so that case keeps its
    original form and simply fails to match, which is the correct outcome.

    Applied to a fixed point, because the two markers occur in **both** orders:
    `丁(姓)_A` puts the variant suffix last, `嘉義_A(地名)` puts it in the middle,
    and an end-anchored suffix rule only catches the first. One pass left 13
    twtsl entries keyed as `嘉義_A`, which no corpus gloss can ever match and
    which nothing would have reported. Idempotence is asserted over the whole
    dictionary by `12_strata.py --audit`.
    """
    if not gloss:
        return ""
    g = gloss.strip()
    for _ in range(4):
        out = _PAREN.sub("", _VARIANT.sub("", g).strip()).strip() or g
        if out == g:
            return g
        g = out
    return g


# ------------------------------------------------------------------ phonology


class Phonology:
    """twtsl's slot-1 (location, handshape) coding, and the neighbourhoods it
    induces over sign types."""

    def __init__(self, db_path: Path | str = TWTSL_DB, lang: str = "zh"):
        import sqlite3

        self.path = Path(db_path)
        con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        sig: dict[tuple[int, int], tuple[str, str]] = {}
        for i, r, loc, hs in con.execute(
            "SELECT id, row_id, location, hs_1 FROM phonology "
            "WHERE lang = ? AND slot = 1", (lang,)
        ):
            if loc and hs:
                sig[(i, r)] = (loc, hs)

        self.by_gloss: dict[str, set[tuple[str, str]]] = {}
        self.clip: dict[str, list[str]] = {}
        for i, r, name, clip in con.execute(
            "SELECT id, row_id, name, clip FROM entries WHERE lang = ?", (lang,)
        ):
            g = normalise_gloss(name)
            if not g:
                continue
            if (i, r) in sig:
                self.by_gloss.setdefault(g, set()).add(sig[(i, r)])
            if clip:
                self.clip.setdefault(g, []).append(clip)

        # Neighbourhood size counts distinct *lexemes*, not entries: two variants
        # of one sign that happen to share a signature are one competitor, not
        # two, and counting entries would inflate every crowded neighbourhood by
        # however many A/B pairs happened to fall in it.
        self.size: dict[tuple[str, str], int] = {}
        for g, sigs in self.by_gloss.items():
            for s in sigs:
                self.size[s] = self.size.get(s, 0) + 1

    def density(self, gloss: str) -> dict | None:
        """Neighbourhood size for a corpus gloss, or None if twtsl lacks it."""
        sigs = self.by_gloss.get(normalise_gloss(gloss))
        if not sigs:
            return None
        sizes = sorted(self.size[s] for s in sigs)
        return {
            "density": sum(sizes) / len(sizes),
            "density_min": sizes[0],
            "density_max": sizes[-1],
            "n_signatures": len(sigs),  # >1 means the form is not pinned down
        }


# --------------------------------------------------------------- reduction

# A sign type needs this many tokens before its median is a usable reference.
# Five is where coverage (67% of tokens) and stability trade off; at three the
# median of three durations is itself mostly noise.
MIN_TOKENS_PER_TYPE = 5


def type_medians(tokens_by_key: dict[str, list[tuple[str, int]]],
                 min_tokens: int = MIN_TOKENS_PER_TYPE) -> dict[str, float]:
    """Median running duration per sign type — the reference each token is read
    against. Median, not mean, because token durations are right-skewed."""
    per: dict[str, list[int]] = {}
    for toks in tokens_by_key.values():
        for gloss, dur in toks:
            g = normalise_gloss(gloss)
            if g and dur > 0:
                per.setdefault(g, []).append(dur)
    out = {}
    for g, v in per.items():
        if len(v) >= min_tokens:
            v.sort()
            out[g] = v[len(v) // 2]
    return out


# ------------------------------------------- citation duration (measured, rejected)
#
# Kept because the module docstring cites its numbers and a rejection nobody can
# reproduce is just an assertion. Used only by `12_strata.py --audit-citation`;
# nothing in the built strata depends on it.

FPS = 15
FRAME_W, FRAME_H = 128, 96
REL_THR = 0.2  # fraction of peak motion that counts as "signing"
MIN_BURST_FRAMES = 3  # below this a run above threshold is camera noise


def motion_envelope(clip: Path, fps: int = FPS) -> list[float]:
    """Mean absolute frame-to-frame difference, one value per frame.

    Greyscale at 128x96 because only the *shape* of the envelope matters and a
    small frame makes this decode-bound rather than compute-bound.
    """
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(clip),
        "-vf", f"fps={fps},scale={FRAME_W}:{FRAME_H},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, timeout=120).stdout
    n = FRAME_W * FRAME_H
    frames = [out[i * n:(i + 1) * n] for i in range(len(out) // n)]
    if len(frames) < 2:
        return []
    import numpy as np

    arr = np.frombuffer(b"".join(frames), np.uint8).reshape(-1, FRAME_H, FRAME_W)
    return np.abs(np.diff(arr.astype(np.int16), axis=0)).mean(axis=(1, 2)).tolist()


def bursts_ms(env: list[float], fps: int = FPS, rel_thr: float = REL_THR,
              min_frames: int = MIN_BURST_FRAMES) -> list[float]:
    """Durations of the separate runs of motion in a clip, in ms.

    Separate runs, not one span: the mean dictionary clip holds 2.2 of them
    because the convention is to produce each sign twice, and a first-to-last
    span would silently measure both productions and the pause between them.
    """
    if not env:
        return []
    peak = max(env)
    if peak <= 0:
        return []
    thr = rel_thr * peak
    runs, cur = [], 0
    for v in list(env) + [0.0]:
        if v >= thr:
            cur += 1
        else:
            if cur >= min_frames:
                runs.append(cur * 1000.0 / fps)
            cur = 0
    return runs


def citation_ms(gloss: str, phon: Phonology, media: Path = TWTSL_MEDIA,
                rel_thr: float = REL_THR) -> float | None:
    """Median single-burst duration across every clip twtsl has for this sign."""
    vals = []
    for c in phon.clip.get(normalise_gloss(gloss)) or []:
        p = media / f"{c}.mp4"
        if not p.exists():
            continue
        b = bursts_ms(motion_envelope(p), rel_thr=rel_thr)
        if b:
            b.sort()
            vals.append(b[len(b) // 2])
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


# ------------------------------------------------------------- record strata

# A record needs this many typed tokens before it gets a class rather than
# "unknown". One matched token out of nine says nothing about the utterance, and
# a stratum that is mostly noise would dilute the very contrast it exists to
# show.
MIN_COVERED = 2


def record_strata(tokens_by_key: dict[str, list[tuple[str, int]]], phon: Phonology,
                  reference: dict[str, float], min_covered: int = MIN_COVERED) -> dict:
    """(gloss, duration_ms) per utterance -> the per-utterance stratum values.

    Aggregation is the mean over the utterance's *covered* tokens, plus the
    coverage itself, so a thinly covered utterance can be excluded rather than
    trusted. `reference` is `type_medians(...)`.
    """
    out: dict[str, dict] = {}
    for key, toks in tokens_by_key.items():
        dens, red, amb = [], [], 0
        for gloss, dur in toks:
            d = phon.density(gloss)
            if d is not None:
                dens.append(d["density"])
                amb += d["n_signatures"] > 1
            ref = reference.get(normalise_gloss(gloss))
            if ref and dur > 0:
                red.append(dur / ref)
        row = {
            "n_tokens": len(toks),
            # Duration of the whole utterance, so the report can ask whether a
            # "reduced utterances are harder" effect is really "short utterances
            # are harder". They are different claims.
            "dur_ms": sum(d for _, d in toks),
            "n_density": len(dens),
            "n_reduction": len(red),
            "ambiguous_variants": amb,
            "density": sum(dens) / len(dens) if dens else None,
            "reduction": sum(red) / len(red) if red else None,
            # Per-token values, for the split-half reliability check. Dropped
            # before the strata are written out — the artefact is per utterance.
            "_rel": red,
        }
        row["density_ok"] = len(dens) >= min_covered
        row["reduction_ok"] = len(red) >= min_covered
        out[key] = row
    return out


# Utterance-length bins the strata are balanced inside. Long utterances contain
# more *common* signs, and common signs sit in crowded neighbourhoods, so an
# unbalanced density tercile is partly a length tercile: measured at 3.2 s / 4.3 s
# / 5.3 s mean duration across the three classes, which is enough to explain a
# difference on its own. Splitting inside length bins removes that by
# construction instead of caveating it afterwards.
LENGTH_BINS = (3, 6, 9)


def _bin(n: int, bins=LENGTH_BINS) -> int:
    for i, b in enumerate(bins):
        if n <= b:
            return i
    return len(bins)


def add_classes(strata: dict, field: str, labels=("low", "mid", "high"),
                within: str | None = "n_tokens", bins=LENGTH_BINS) -> dict:
    """Tercile classes over the records that have the measure.

    Terciles rather than absolute cut-offs: both measures are on scales that
    depend on decisions made in this module (which slot, which reference), and a
    rank-based split is invariant to all of them.

    `within` names a covariate to balance on: terciles are taken *inside* each
    bin of it, so the classes cannot differ systematically in that covariate.
    Pass None for plain global terciles. Cut points are returned per bin so the
    paper can state them.
    """
    ok = [(k, r) for k, r in strata.items()
          if r.get(field) is not None and r.get(f"{field}_ok")]
    for r in strata.values():
        r[f"{field}_class"] = None
    if len(ok) < 3:
        return {"field": field, "within": within, "bins": [], "n": len(ok)}

    groups: dict[int, list] = {}
    for k, r in ok:
        groups.setdefault(_bin(r[within], bins) if within else 0, []).append((k, r))

    out = []
    for b, members in sorted(groups.items()):
        vals = sorted(r[field] for _, r in members)
        if len(vals) < 3:
            continue  # too thin to split; those records stay unclassed
        lo, hi = vals[len(vals) // 3], vals[2 * len(vals) // 3]
        for _, r in members:
            v = r[field]
            r[f"{field}_class"] = (labels[0] if v < lo
                                   else labels[2] if v >= hi else labels[1])
        out.append({"bin": b, "cuts": [lo, hi], "n": len(members)})
    return {"field": field, "within": within, "bins": out, "n": len(ok)}
