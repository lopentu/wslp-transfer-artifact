"""Metrics.

The primary number is **contrastive referent accuracy**, split by whether the item
needs discourse at all:

    anaphoric  — the pointing sign is anchored to a locus set earlier
    deictic    — the pointing sign is anchored to someone physically present

and the headline quantities are differences between conditions:

    context benefit      = correct  - local
    context selectivity  = correct  - mismatched
    context robustness   = local    - irrelevant/shuffled

The interesting comparison is not any single accuracy but the *interaction*:
benefit(anaphoric) - benefit(deictic). A model whose context gain is really
target-side fluency shows no interaction, because fluency helps both. A model that
grounds reference shows benefit concentrated on the anaphoric half and
selectivity > 0 there.

Free-translation scores (BLEU / chrF) are secondary and reported with the zh
tokenizer, because Chinese word segmentation would otherwise dominate the number.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


# ------------------------------------------------------------------ contrastive


def contrastive_accuracy(scores: list[list[dict]], key: str = "mean") -> list[bool]:
    """Gold is candidates[0], so a hit is argmax == 0."""
    out = []
    for s in scores:
        if len(s) < 2:
            out.append(False)
            continue
        vals = [c[key] for c in s]
        out.append(max(range(len(vals)), key=lambda i: vals[i]) == 0)
    return out


def choice(scores: list[list[dict]], key: str = "mean") -> list[int]:
    """Which candidate the model picked, gold being index 0.

    `contrastive_accuracy` collapses this to gold/not-gold, which is enough for a
    score and not enough for an agreement figure: three systems can share a
    hit/miss pattern while picking different wrong candidates. -1 marks an item
    with nothing to choose between.
    """
    out = []
    for s in scores:
        if len(s) < 2:
            out.append(-1)
            continue
        vals = [c[key] for c in s]
        out.append(max(range(len(vals)), key=lambda i: vals[i]))
    return out


def margin(scores: list[list[dict]], key: str = "mean") -> list[float]:
    """Gold log-prob minus the best distractor's — how decisively, not just whether."""
    out = []
    for s in scores:
        if len(s) < 2:
            out.append(float("nan"))
            continue
        vals = [c[key] for c in s]
        out.append(vals[0] - max(vals[1:]))
    return out


# ----------------------------------------------------------------- pronoun slot


def pronoun_slot_hit(hyp: str, gold_gloss: str, alternates: list[str]) -> bool | None:
    """Did free generation put the right person/number in?

    None when the hypothesis commits to neither the gold nor any alternate — the
    item is then simply not answered, and counting it as wrong would conflate
    "chose the wrong referent" with "did not translate the pronoun at all".
    """
    if gold_gloss and gold_gloss in hyp:
        return True
    if any(a in hyp for a in alternates):
        return False
    return None


# ----------------------------------------------------------- aggregation & CIs


@dataclass
class Cell:
    n: int
    acc: float
    lo: float
    hi: float
    mean_margin: float

    def fmt(self) -> str:
        return f"{self.acc:5.1%} [{self.lo:.1%},{self.hi:.1%}] n={self.n}"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval — the right one at these n (65 anaphoric items per fold)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def cell(hits: list[bool], margins: list[float] | None = None) -> Cell:
    n = len(hits)
    k = sum(hits)
    lo, hi = wilson(k, n)
    mm = float("nan")
    if margins:
        good = [m for m in margins if not math.isnan(m)]
        mm = sum(good) / len(good) if good else float("nan")
    return Cell(n, k / n if n else 0.0, lo, hi, mm)


def by_group(rows: list[dict], group: str, hit_key: str = "hit") -> dict[str, Cell]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[str(r.get(group))].append(r)
    return {
        g: cell([r[hit_key] for r in v], [r.get("margin", float("nan")) for r in v])
        for g, v in sorted(buckets.items())
    }


def paired_delta(a: list[dict], b: list[dict], hit_key: str = "hit") -> dict:
    """Condition A minus condition B on the *same* items, with a sign test.

    Paired throughout: both conditions score the identical item set, so the only
    thing varying is the context supplied. That also makes the McNemar-style
    counts meaningful — n_ab is items A gets and B misses.
    """
    ia = {r["key"]: r[hit_key] for r in a}
    ib = {r["key"]: r[hit_key] for r in b}
    keys = sorted(set(ia) & set(ib))
    n_ab = sum(1 for k in keys if ia[k] and not ib[k])
    n_ba = sum(1 for k in keys if ib[k] and not ia[k])
    delta = (sum(ia[k] for k in keys) - sum(ib[k] for k in keys)) / max(1, len(keys))
    return {
        "n": len(keys),
        "delta": delta,
        "n_gain": n_ab,
        "n_loss": n_ba,
        "p_exact": _sign_test(n_ab, n_ba),
    }


def _sign_test(a: int, b: int) -> float:
    """Two-sided exact binomial on the discordant pairs (McNemar, exact form)."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def interaction(bene_ana: dict, bene_dei: dict) -> dict:
    """benefit(anaphoric) - benefit(deictic).

    This is the paper's key statistic. It is a difference of two independent
    paired deltas, so the variance adds; the interval below is normal-approximate
    on the two per-item difference distributions and should be quoted as
    approximate.
    """
    d = bene_ana["delta"] - bene_dei["delta"]
    va = bene_ana.get("var", 0.0)
    vd = bene_dei.get("var", 0.0)
    se = math.sqrt(va + vd) if (va or vd) else float("nan")
    return {"interaction": d, "se": se,
            "ci": (d - 1.96 * se, d + 1.96 * se) if se == se else None}


def delta_var(a: list[dict], b: list[dict], hit_key: str = "hit") -> float:
    """Variance of the paired per-item difference, for the interaction interval."""
    ia = {r["key"]: float(r[hit_key]) for r in a}
    ib = {r["key"]: float(r[hit_key]) for r in b}
    keys = sorted(set(ia) & set(ib))
    n = len(keys)
    if n < 2:
        return 0.0
    d = [ia[k] - ib[k] for k in keys]
    m = sum(d) / n
    return sum((x - m) ** 2 for x in d) / (n - 1) / n


# -------------------------------------------------------------------- secondary


def _to_traditional(texts: list[str]) -> list[str] | None:
    """Simplified -> traditional, or None if no converter is installed."""
    try:
        import opencc
    except ImportError:
        return None
    conv = opencc.OpenCC("s2t")
    return [conv.convert(t) for t in texts]


def corpus_bleu_chrf(hyps: list[str], refs: list[str]) -> dict:
    """BLEU/chrF, or NaN if sacrebleu is absent.

    Degrades rather than raising: these are the paper's secondary metrics, and a
    missing optional dependency should not take the contrastive tables down with
    it.

    Reported twice, raw and after normalising both sides to traditional
    characters, because the two are not close. The CSL checkpoints were
    pretrained to emit **simplified** Chinese and this corpus's targets are
    **traditional**, so a residual script preference is scored as if it were a
    translation error: on the `zh` tokeniser a sentence that is correct but in
    the wrong script scores BLEU 14 where the identical traditional string
    scores 100. Raw BLEU alone would therefore penalise precisely the rows whose
    transfer we are trying to measure, and a CSL-over-ASL gap read off it would
    partly be a measure of script conversion. `*_norm` is NaN when `opencc` is
    absent, which is a missing number rather than a silently un-normalised one.
    """
    if not hyps:
        return {"bleu": 0.0, "chrf": 0.0, "bleu_norm": 0.0, "chrf_norm": 0.0}
    try:
        import sacrebleu
    except ImportError:
        return {k: float("nan")
                for k in ("bleu", "chrf", "bleu_norm", "chrf_norm")}
    out = {
        "bleu": sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh").score,
        "chrf": sacrebleu.corpus_chrf(hyps, [refs]).score,
    }
    h, r = _to_traditional(hyps), _to_traditional(refs)
    if h is None:
        out["bleu_norm"] = out["chrf_norm"] = float("nan")
    else:
        out["bleu_norm"] = sacrebleu.corpus_bleu(h, [r], tokenize="zh").score
        out["chrf_norm"] = sacrebleu.corpus_chrf(h, [r]).score
    return out
