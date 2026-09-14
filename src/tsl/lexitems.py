"""The lexical contrastive diagnostic — a second dependent variable, built to
answer the one objection the pronoun diagnostic cannot.

`items.py` builds a 4-way forced choice over the person/number of a **pointing
sign**. Measured on 2026-08-14, every one of its 975 items does: the gold gloss
is 我 in 528 of them and a 1st/2nd-person point in 785. That is a problem for the
paper's headline, not a cosmetic one.

Pointing is the most cross-linguistically uniform device sign languages have.
First person is an index point to the signer's own chest in TSL, in CSL and in
ASL alike; second person points at the addressee; third person points at a locus
in the signing space. None of it is lexically arbitrary, so none of it requires
language-specific lexical knowledge to read. A diagnostic made entirely of
pointing therefore **cannot distinguish** two accounts of our central null
result:

* the honest one — an encoder's *source sign language* does not constrain what it
  transfers, so an ASL encoder serves a Chinese decoder as well as a CSL one;
* the confounded one — we happened to measure the single construction that ASL
  and CSL already share, and would have seen a difference anywhere else.

Both predict everything observed: that the encoder must be pretrained (reading
continuous video at all is language-neutral), that its sign language does not
matter (the target construction is shared), and that the decoder's written
language carries the spread (the output is Chinese). Note that only the *null*
half is at risk. The decoder effect is about which language is emitted and would
appear on any Chinese-output measure.

So this module builds the same forced choice over a **content word** instead.
The distractors are TSL signs with a demonstrably different form, so the answer
is available in the video and nowhere else, and a content sign is lexically
arbitrary — there is no reason for TSL 工作 and ASL WORK to resemble one another.
If the source sign language still does not matter here, the general claim is
earned. If it starts to matter, the result is sharper than the one we have: the
decoder transfers generally, the encoder only over the shared indexical
substrate.

Five constraints, each because dropping it would let something other than the
video answer the item:

1. **The gold gloss must appear verbatim in the Chinese translation, exactly
   once.** This is what makes the item honest, and it does more work than it
   looks. The gloss is the annotator's record of the sign that was *actually
   produced*; requiring it to surface in the translation guarantees the word we
   swap corresponds to a real sign in the clip, rather than to a translator's
   paraphrase of something never signed.
2. **Distractors are drawn from disjoint slot-1 (location, handshape)
   signatures.** Two signs that differ in both location and handshape are
   visually distinct at the coarsest level twtsl codes, so the clip determines
   which was produced. It also rules out the nastiest failure mode: a
   near-synonym that is the *same sign*, which would make the item unanswerable
   in principle rather than hard.
3. **Only glosses with a single twtsl signature are eligible**, as gold or as
   distractor. A gloss with disagreeing variants (`會_N` at the back of the hand,
   `會_S` at the temple) is not pinned down to one form, so "visually distinct
   from gold" would be unverifiable for it — see `strata.normalise_gloss`.
4. **Distractors are frequency-matched to gold.** Pointing glosses are steeply
   imbalanced and `items.mark_frequency_prior` has to correct for it after the
   fact; here the imbalance is designed out, by drawing distractors from gold's
   own log-frequency neighbourhood. `prior_correct` is still recorded so the
   correction remains available and the claim is checkable rather than asserted.
5. **One item per utterance.** Two items sharing a clip are not two independent
   observations, and pooling them would shrink every confidence interval on the
   strength of video that was only seen once.

What this module deliberately does **not** control is semantic plausibility: a
frequency-matched, visually-distinct distractor can still be one the preceding
Chinese makes obviously wrong. That is not assumed away — it is measured, by
running `07_shortcuts.py` over these items and reporting the text-hard subset,
exactly as the pronoun set does. Selecting distractors an LLM finds plausible was
considered and rejected for this round: it would tune the items against one
particular language model, and the subset machinery already bounds the leak
without that circularity.

## Iconicity, and why it is a stratum rather than a caveat

Replacing pointing with content words weakens the confound but does not remove
it, because lexical signs are not uniformly arbitrary. An **iconic** sign is
motivated by resemblance to its referent, so unrelated sign languages converge on
it independently: TSL and ASL both mime a steering wheel for 開車 and both trace
a swimming fish for 魚, with no shared history required. A content-word
diagnostic made entirely of iconic signs would reproduce the pointing problem one
level down.

So iconicity is *measured per gloss and reported as a stratum*, from **ASL-LEX
2.0's human iconicity ratings** — see `asllex.py`, which also records why the
SignCLIP-derived alternative was built, tested and rejected (it correlates +0.10
with these ratings, and its per-gloss reliability cannot be computed at all
because only 4 of 3,428 glosses have more than one clip). The SignCLIP figures
are still attached to each record as `asl_z`/`csl_z` so the measurement stays
inspectable, but they are secondary and must not be used to stratify.

**Keep this stratum in its place.** It is a basic composition check — evidence
that the item set spans the iconicity range rather than sitting entirely at the
convergent end — and it is peripheral to the argument. It is *not* the paper's
headline test, and the write-up should not lean on a tercile gradient: the
instrument is a proxy two translation joins removed from the quantity of
interest, it reaches only 57% of the items, and the concepts it misses are the
frequent ones. Reporting it as a dose-response would claim far more than it can
carry.

State the stratifier precisely or not at all. `Iconicity(M)` is how iconic raters
judged the *ASL* sign to be; it is a proxy for which concepts invite unrelated
sign languages to converge, **not** a measurement of TSL-ASL form similarity.
"""

from __future__ import annotations

import math
import random
import re
import zlib
from collections import Counter
from pathlib import Path

from .asllex import iconicity_by_zh
from .db import ANAPHORIC, DEICTIC, Corpus
from .items import Record, build_records
from .strata import Phonology, normalise_gloss

# Exact tags only. The corpus also emits compound tags (`動詞+名詞`,
# `代形詞+指示詞`) where one annotation covers two signs; which half a given
# character run belongs to is not recoverable, so those slots are skipped rather
# than guessed.
CONTENT_POS = ("名詞", "動詞", "形容詞")

PRONOUNS = DEICTIC | ANAPHORIC

# A gold or distractor type needs this many corpus tokens before it is used. Not
# a statistical threshold — a hapax gloss is disproportionately likely to be an
# annotation slip, and a distractor nobody ever signs makes the item easy for the
# wrong reason.
MIN_FREQ = 3

# Distractors are drawn from the `POOL_NEAR` types nearest gold in log frequency,
# then three are chosen by an RNG seeded on the item key. Taking the nearest
# three outright would hand the same trio to every gold in a frequency band and
# turn "these words are always the wrong answer" into a learnable cue.
#
# Forty rather than a dozen because the pool is really the *search space* for the
# fluency matching in `17_lm_match_distractors.py`, not the final choice. At
# twelve, only 38% of items could field three distractors within 0.35 nats of
# gold, and the survivors would have been a biased sample — the items whose
# frequency band happened to contain fluent substitutes. Widening the band trades
# exact frequency matching for fluency matching, which is the tighter constraint;
# `prior_correct` is recomputed after the choice so the frequency prior remains
# reportable rather than assumed.
POOL_NEAR = 40
N_DISTRACTORS = 3


def _coarse_pos(pos: str) -> str | None:
    return pos if pos in CONTENT_POS else None


class _Superstrings:
    """Longer vocabulary items that contain a given word.

    Substituting a word into a translation is a plain string replace, so a gold
    that happens to sit inside a longer word corrupts the sentence instead of
    changing it: 人 inside 聾人 would rewrite 聾人朋友 to 聾X朋友 and the item
    would be scored on a candidate set that is nonsense in all four positions.
    `text.count(word) == 1` does not catch this — the count is 1 precisely
    because the only occurrence is the embedded one.
    """

    def __init__(self, vocab: set[str]):
        self.vocab = sorted(vocab)
        self._cache: dict[str, list[str]] = {}

    def of(self, word: str) -> list[str]:
        hit = self._cache.get(word)
        if hit is None:
            hit = [v for v in self.vocab if len(v) > len(word) and word in v]
            self._cache[word] = hit
        return hit


class _Segmenter:
    """Word boundaries in the *translation*, which the gloss vocabulary misses.

    The superstring guard alone is not enough, because it can only block words it
    has seen. The translation is Chinese prose, not a gloss sequence, so it
    contains words no annotator ever wrote: the gloss 改 surfaces inside the
    common verb 改進, which is in neither the corpus token inventory nor
    twtsl, and substituting the gloss inside it left a word fragment — a candidate set
    that is broken in three of its four positions and would have been scored
    anyway.

    jieba's dictionary is Simplified, so the text is converted with OpenCC,
    segmented, and converted back. The round trip is safe for this purpose
    because both directions are character-for-character here — asserted per call
    on the length, and the slot is refused rather than trusted if it ever is not.
    """

    def __init__(self):
        import jieba
        import opencc

        jieba.setLogLevel(60)
        self._cut = jieba.lcut
        self._t2s = opencc.OpenCC("t2s").convert
        self._cache: dict[str, set[str] | None] = {}

    def tokens(self, text: str) -> set[str] | None:
        """Segment offsets as (start, end) pairs, or None if unusable."""
        hit = self._cache.get(text, ...)
        if hit is not ...:
            return hit
        sim = self._t2s(text)
        spans: set[tuple[int, int]] | None
        if len(sim) != len(text):
            spans = None  # conversion moved the offsets; do not guess
        else:
            spans, i = set(), 0
            for w in self._cut(sim):
                spans.add((i, i + len(w)))
                i += len(w)
        self._cache[text] = spans
        return spans


# A question mark ends a clause; one sitting immediately before a Chinese
# character is a character the transcription lost (一杯?熱, 開?視訊, 十五?六),
# as are the usual replacement glyphs.
_BROKEN = re.compile(r"[�□]|[?？](?=[一-鿿])")


def text_is_clean(text: str) -> bool:
    """Is the translation free of characters the transcription dropped?

    Such a gap sits identically in the gold and in all three distractors, so it
    biases no candidate and the item is not *wrong*. It is dropped anyway: three
    items are worth less than a reviewer opening the released item file and
    finding mojibake in it.
    """
    return not _BROKEN.search(text)


def _slot_is_safe(word: str, text: str, sent_words: set[str],
                  supers: _Superstrings, seg: _Segmenter) -> bool:
    """Can `word` be replaced in `text` without damaging a neighbouring word?"""
    if text.count(word) != 1:
        return False
    if any(word != u and word in u for u in sent_words):
        return False
    lo = text.index(word)
    hi = lo + len(word)
    spans = seg.tokens(text)
    if spans is None or (lo, hi) not in spans:
        return False  # not a word of the translation on its own
    for v in supers.of(word):
        start = 0
        while (j := text.find(v, start)) != -1:
            if j < hi and j + len(v) > lo:
                return False  # the run is part of a longer word
            start = j + 1
    return True


# ------------------------------------------------------- cross-language form

SIGNCLIP_EMB = Path("data/signclip/embeddings.npz")


def form_convergence(path: Path | str = SIGNCLIP_EMB) -> dict[str, dict]:
    """Per-gloss z-scored closeness of each sign language's form to TSL's.

    Returns `{gloss: {"ase": z, "csl": z, "n_clips": k}}`. The z-score is taken
    **across the 41 languages within one gloss**, so it answers "is ASL unusually
    close to TSL for this concept" and not "is this concept easy to embed", which
    is a property of the concept and varies far more. Missing file returns `{}`:
    the item set is still buildable without the stratifier, just not stratifiable.
    """
    import numpy as np

    p = Path(path)
    if not p.exists():
        return {}
    z = np.load(p, allow_pickle=True)
    langs = [str(x) for x in z["langs"]]
    zh = [str(x) for x in z["zh"]]
    gi = z["gloss_index"]
    V = z["video"]
    T = z["text"]
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    T = T / np.linalg.norm(T, axis=2, keepdims=True)

    ok = gi >= 0
    # (clips, languages): every language's text for each clip's own concept
    sims = np.full((len(zh), len(langs)), np.nan, np.float32)
    sims[ok] = np.einsum("cd,lcd->cl", V[ok], T[:, gi[ok], :])
    mu = sims.mean(1, keepdims=True)
    sd = sims.std(1, keepdims=True)
    zs = (sims - mu) / np.where(sd > 0, sd, np.nan)

    per: dict[str, list] = {}
    for w, row, good in zip(zh, zs, ok):
        if good:
            per.setdefault(w, []).append(row)
    idx = {c: langs.index(c) for c in ("ase", "csl") if c in langs}
    out = {}
    for w, rows in per.items():
        m = np.nanmean(np.stack(rows), 0)
        out[w] = {c: float(m[i]) for c, i in idx.items()}
        out[w]["n_clips"] = len(rows)
    return out


def _eligible_types(phon: Phonology, freq: Counter) -> dict[str, dict]:
    """Types usable as gold or distractor, keyed by surface form.

    Eligibility is a property of the *type*, so gold and distractor are held to
    one standard and an item can never contrast a pinned-down form against an
    ambiguous one.
    """
    out: dict[str, dict] = {}
    for w, n in freq.items():
        if n < MIN_FREQ or w in PRONOUNS:
            continue
        sigs = phon.by_gloss.get(normalise_gloss(w))
        if not sigs or len(sigs) != 1:
            continue  # absent from twtsl, or its variants disagree on the form
        d = phon.density(w)
        out[w] = {
            "sig": next(iter(sigs)),
            "freq": n,
            "logf": math.log(n),
            "density": None if d is None else d["density"],
        }
    return out


def _distractors(gold: str, pos: str, key: str, pool: list[tuple[str, dict]],
                 types: dict[str, dict], blocked: set[str]
                 ) -> tuple[list[str], list[str]] | None:
    """Frequency-matched types whose sign is visually distinct from gold.

    Returns `(chosen, band)`. The band is the whole frequency-matched pool and is
    kept on the record so `17_lm_match_distractors.py` can re-choose from it
    without rebuilding: this stage cannot tell which distractors leave a fluent
    sentence, and that decision needs a language model rather than a lexicon.
    """
    g = types[gold]
    near = [w for w, t in pool
            if w != gold and w not in blocked and t["sig"] != g["sig"]]
    if len(near) < N_DISTRACTORS:
        return None
    near.sort(key=lambda w: (abs(types[w]["logf"] - g["logf"]), w))
    band = near[:POOL_NEAR]
    # crc32, not hash(): the builtin string hash is salted per process, so the
    # item set would differ between builds and stop being the same instrument.
    rng = random.Random(zlib.crc32(f"{key}|{gold}|{pos}".encode()))
    return sorted(rng.sample(band, N_DISTRACTORS)), band


def build_lexical_records(corpus: Corpus | None = None, ctx_k: int = 8,
                          db_path: Path | str | None = None) -> list[dict]:
    """Records carrying a lexical contrastive item where one could be built.

    The base records — spans, context, text — come from `items.build_records`, so
    the video and context plumbing is shared with the pronoun set by construction
    and no difference between the two diagnostics can be a difference in how the
    clip was cut. Only the diagnostic fields are replaced.
    """
    corpus = corpus or Corpus()
    phon = Phonology(db_path) if db_path else Phonology()
    para = corpus.by_paragraph()
    base: list[Record] = build_records(corpus, ctx_k)

    sent_by_key = {s.key: s for sents in para.values() for s in sents}

    freq: Counter = Counter()
    for s in sent_by_key.values():
        for _, w, pos in s.tokens:
            if _coarse_pos(pos):
                freq[w] += 1

    types = _eligible_types(phon, freq)
    by_pos: dict[str, Counter] = {p: Counter() for p in CONTENT_POS}
    for s in sent_by_key.values():
        for _, w, pos in s.tokens:
            if _coarse_pos(pos) and w in types:
                by_pos[pos][w] += 1
    pools = {p: [(w, types[w]) for w in c] for p, c in by_pos.items()}

    vocab = set(freq) | {normalise_gloss(g) for g in phon.by_gloss}
    supers = _Superstrings({v for v in vocab if v})
    seg = _Segmenter()
    conv = form_convergence()   # secondary, inspectable, never the stratifier
    icon = iconicity_by_zh()

    out: list[dict] = []
    for r in base:
        d = r.as_dict()
        # Start from a clean slate: these records must never be mistaken for
        # pronoun items, and a leftover `item_type` would silently mix the two
        # diagnostics inside one eval run.
        d.update({"item_type": None, "pron_gloss": None, "pron_idx": None,
                  "resolved_referent": False, "candidates": [],
                  "prior_correct": None, "ctx_dist": None, "n_candidates": 0})

        s = sent_by_key.get(r.key)
        if s is not None and text_is_clean(r.text):
            words = {w for _, w, _ in s.tokens}
            ctx_text = " ".join(c["text"] for c in r.ctx)
            slots = []
            for idx, w, pos in sorted(s.tokens):
                if _coarse_pos(pos) is None or w not in types:
                    continue
                if not _slot_is_safe(w, r.text, words, supers, seg):
                    continue
                slots.append((idx, w, pos))
            if slots:
                rng = random.Random(zlib.crc32(r.key.encode()))
                idx, w, pos = rng.choice(slots)
                blocked = {x for x in types if x in r.text or x in ctx_text}
                picked = _distractors(w, pos, r.key, pools[pos], types, blocked)
                if picked:
                    alts, band = picked
                    cands = [r.text] + [r.text.replace(w, a, 1) for a in alts]
                    best = max([w] + alts, key=lambda x: (types[x]["freq"], x))
                    cv = conv.get(w, {})
                    ic = icon.get(w, {})
                    d.update({
                        "item_type": "lexical",
                        "pron_idx": idx,
                        "candidates": cands,
                        "prior_correct": best == w,
                        "gold_word": w,
                        "gold_pos": pos,
                        "distractors": alts,
                        "gold_freq": types[w]["freq"],
                        "gold_density": types[w]["density"],
                        "n_slots": len(slots),
                        "distractor_pool": band,
                        # Parallel to distractor_pool, so `prior_correct` can be
                        # recomputed downstream without reloading the corpus.
                        "pool_freq": [types[x]["freq"] for x in band],
                        # The stratifier: human iconicity rating (1-7) of the ASL
                        # sign for this concept. None where the concept does not
                        # reach ASL-LEX through the two translation joins; those
                        # items form a fourth, unstratified group rather than
                        # being dropped or counted as low-iconicity.
                        "icon": ic.get("icon"),
                        "icon_en": ic.get("en"),
                        "icon_n_en": ic.get("n_en"),
                        # Secondary, and not to be reported as form similarity:
                        # the SignCLIP z-scores this stratifier replaced.
                        "asl_z": cv.get("ase"),
                        "csl_z": cv.get("csl"),
                    })
        out.append(d)
    return out


def add_iconicity_class(records: list[dict], field: str = "icon",
                        out: str = "icon_class") -> list[float]:
    """Tercile the items by the gold concept's iconicity, in place.

    Terciles over the *item* distribution rather than the type distribution: the
    stratum has to be balanced in the thing that gets scored, and a gloss
    contributing seventy items would otherwise count once. Items whose gold does
    not reach ASL-LEX get `None` and are reported separately — an unratable item
    is not a low-iconicity one, and folding the two together would put every
    culture-specific concept the join missed into exactly the stratum the
    hypothesis cares about.
    """
    items = [r for r in records if r.get("item_type") == "lexical"]
    for r in items:
        r[out] = None
    vals = sorted(r[field] for r in items if r.get(field) is not None)
    if len(vals) < 3:
        return []
    lo, hi = vals[len(vals) // 3], vals[2 * len(vals) // 3]
    for r in items:
        v = r.get(field)
        if v is not None:
            r[out] = "low" if v < lo else "high" if v >= hi else "mid"
    return [lo, hi]


def summarise(records: list[dict]) -> dict:
    items = [r for r in records if r.get("item_type") == "lexical"]
    c: Counter = Counter()
    c["records"] = len(records)
    c["items"] = len(items)
    c["items_with_context"] = sum(1 for r in items if r["ctx"])
    c["prior_would_answer"] = sum(1 for r in items if r["prior_correct"])
    for r in items:
        c[f"pos:{r['gold_pos']}"] += 1
        c[f"len:{len(r['gold_word'])}"] += 1
        c[f"icon:{r.get('icon_class')}"] += 1
    return dict(c)
