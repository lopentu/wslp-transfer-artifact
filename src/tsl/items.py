"""Build the training records and the referent-sensitivity diagnostic items.

The diagnostic needs no new annotation. It exploits a split that the corpus
already carries: a pointing sign glossed 我/你 is anchored to a participant who
is physically present, so the target clip alone determines it; a pointing sign
glossed 他/她/他們 is anchored to a locus assigned earlier in the discourse, so
the target clip alone does *not* determine it.

That gives a within-corpus control. A model that genuinely grounds discourse
should gain from correct context on the anaphoric items and gain ~nothing on the
deictic ones. A model whose "context gain" is really target-side fluency gains
equally on both — and loses nothing when the context is swapped for a mismatched
one.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from .db import ANAPHORIC, DEICTIC, Corpus, Sentence, agreeing_verbs

NOUN = "名詞"
PRON = "代名詞"

# Alternates for the person/number contrast: three per gloss, so every item is a
# 4-way choice and chance is 25% for deictic and anaphoric alike.
#
# Two deliberate omissions. 他/她 and 你/妳 differ only in Chinese orthography — the
# TSL pointing sign is identical — so they are never each other's distractors. An
# item asking a model to choose between 他 and 她 from video would be unanswerable
# in principle, and getting it wrong would say nothing about discourse grounding.
# Number contrasts (我/我們, 你/你們) *are* visible, as a sweep rather than a point,
# so those are kept.
ALTERNATES = {
    "我": ["你", "他", "他們"],
    "你": ["我", "他", "他們"],
    "妳": ["我", "他", "她們"],
    "我們": ["我", "你們", "他們"],
    "你們": ["你", "我們", "他們"],
    "他": ["我", "你", "他們"],
    "她": ["我", "你", "她們"],
    "他們": ["我", "你", "他"],
    "她們": ["我", "你", "她"],
    "牠": ["我", "你", "牠們"],
}


@dataclass
class Record:
    """One target utterance, plus everything needed to build every condition."""

    key: str
    uuid: str
    seq: int
    t1: int
    t2: int
    text: str
    speaker: str | None
    para_type: str
    theme: str
    n_tokens: int
    # context: preceding utterances in the same paragraph, nearest last
    ctx: list[dict] = field(default_factory=list)
    # diagnostic
    item_type: str | None = None  # 'deictic' | 'anaphoric'
    pron_gloss: str | None = None
    pron_idx: int | None = None
    resolved_referent: bool = False
    agreeing_verb: str | None = None
    candidates: list[str] = field(default_factory=list)  # candidates[0] is gold
    # True if picking the marginally commonest pointing gloss among this item's
    # candidates would answer it. Set by mark_frequency_prior(); used to define the
    # prior-adversarial subset, never to score.
    prior_correct: bool | None = None
    # Distance in utterances to the nearest preceding person expression, and how
    # many distinct ones the paragraph has introduced so far. Heuristic — see
    # mark_antecedent_distance(). None = no candidate found in the paragraph.
    ctx_dist: int | None = None
    n_candidates: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _first_referring_token(toks: list[tuple[int, str, str]]) -> tuple[int, str, str] | None:
    """The first token tagged 名詞 or 代名詞 in the utterance."""
    for idx, word, pos in sorted(toks):
        if pos.startswith(NOUN) or pos.startswith(PRON):
            return (idx, word, pos)
    return None


def build_records(corpus: Corpus, ctx_k: int = 8) -> list[Record]:
    """One record per sentence, carrying up to `ctx_k` preceding utterances.

    Store generously and select at run time. Measured against a person-NP
    heuristic, the nearest candidate antecedent of an anaphoric item sits a median
    of 2 utterances back, but a 2-utterance window reaches it only 58% of the
    time; 4 reaches 75%, and the whole preceding paragraph 94%. A window too short
    to contain the antecedent guarantees no context can help, which would dilute
    the effect the paper is trying to measure. Keeping 8 here lets
    `TSLDataset(ctx_k=...)` sweep the window without rebuilding the dataset, which
    turns the choice into the context-distance ablation (plan §11, priority 5).
    """
    agree = agreeing_verbs()
    out: list[Record] = []
    for uuid, sents in corpus.by_paragraph().items():
        for i, s in enumerate(sents):
            span = corpus.clamp(uuid, s.t1, s.t2)
            if span is None or not s.text:
                continue  # G4C26's tail: annotated but no video
            ctx = []
            for prev in sents[max(0, i - ctx_k) : i]:
                pspan = corpus.clamp(uuid, prev.t1, prev.t2)
                if pspan is None:
                    continue
                ctx.append(
                    {
                        "seq": prev.seq,
                        "t1": pspan[0],
                        "t2": pspan[1],
                        "text": prev.text,
                        "speaker": prev.speaker,
                        "same_speaker": prev.speaker == s.speaker,
                    }
                )
            r = Record(
                key=s.key,
                uuid=uuid,
                seq=s.seq,
                t1=span[0],
                t2=span[1],
                text=s.text,
                speaker=s.speaker,
                para_type=s.para_type,
                theme=s.theme,
                n_tokens=len(s.tokens),
                ctx=ctx,
            )
            _tag(r, s, agree)
            out.append(r)
    mark_frequency_prior(out)
    mark_antecedent_distance(out)
    return out


def mark_frequency_prior(records: list[Record]) -> Counter:
    """Flag items a marginal-frequency guesser would get right.

    Pointing glosses are steeply imbalanced — 我 alone is two thirds of the deictic
    items — so "always answer 我" scores ~74% on the deictic subset and ~2% on the
    anaphoric one. Comparing raw accuracy across the two subsets would therefore
    be comparing two different prior difficulties, and the deictic control would
    look strong for a reason that has nothing to do with seeing the video.

    So each item records whether the prior answers it, and the report presents the
    interaction on the **prior-adversarial** subset, where the prior is wrong on
    both sides by construction. Frequency is computed corpus-wide rather than
    per-fold on purpose: the subset must denote the same items in every fold.
    """
    freq = Counter(r.pron_gloss for r in records if r.pron_gloss and r.candidates)
    for r in records:
        if not (r.candidates and r.pron_gloss):
            continue
        pool = [r.pron_gloss] + ALTERNATES.get(r.pron_gloss, [])
        best = max(pool, key=lambda w: (freq.get(w, 0), w))
        r.prior_correct = best == r.pron_gloss
    return freq


def _tag(r: Record, s: Sentence, agree: set[str]) -> None:
    """Attach diagnostic labels and contrastive candidates in place."""
    av = next((w for _, w, pos in s.tokens if "動詞" in pos and w in agree), None)
    r.agreeing_verb = av

    if not r.ctx:
        return  # nothing to condition on; usable for training, not as an item

    first = _first_referring_token(s.tokens)
    if first is None:
        return
    idx, word, pos = first
    if not pos.startswith(PRON):
        return  # a lexical NP comes first: reference is fixed inside the clip

    if word in DEICTIC:
        kind = "deictic"
    elif word in ANAPHORIC:
        kind = "anaphoric"
    else:
        return  # 自己 / 大家 / 什麼: neither

    if word not in r.text:
        # The annotator resolved the pointing sign into an explicit NP instead of
        # a pronoun. Informative, but scored separately — the surface form of the
        # answer differs, so it is not comparable to the person-contrast items.
        r.item_type = kind
        r.pron_gloss = word
        r.pron_idx = idx
        r.resolved_referent = True
        return

    if r.text.count(word) != 1:
        return  # ambiguous which occurrence the token licenses
    if word not in ALTERNATES:
        # No distractor set defined (我們兩個, 他們倆個 — the rare compounds). Marking
        # these as items with a one-element candidate list would score them wrong
        # by construction, so they are left out of the contrastive set entirely.
        return

    r.item_type = kind
    r.pron_gloss = word
    r.pron_idx = idx
    r.candidates = [r.text] + [r.text.replace(word, alt, 1) for alt in ALTERNATES[word]]


# ------------------------------------------------------- antecedent candidates

# Person-denoting expressions in the Chinese translations. A heuristic, not an
# annotation layer — the corpus's 回指 layer (the spec's entire 篇章層面) is not
# exposed by the API at all, so there is no gold coreference to be had. Used for
# two things only: measuring how far back the nearest candidate antecedent sits,
# and assembling the candidate set for a referent-resolution task. Never for
# scoring.
PERSON = re.compile(
    r"(聾人|聽人|聽障者?|身障者?|朋友|老師|醫生|護士|同學|同事|學生|老闆|爸爸|媽媽"
    r"|父親|母親|哥哥|弟弟|姊姊|妹妹|先生|太太|女兒|兒子|孩子|小孩|大人|家人|親戚"
    r"|鄰居|客人|店員|司機|警察|翻譯員?|畫家|藝人|對方"
    r"|[一-鿿]{1,3}(?:人員|人們|者|師|生|員|家|工|長|友))"
)


def antecedent_candidates(text: str) -> list[str]:
    seen: list[str] = []
    for m in PERSON.findall(text):
        w = m if isinstance(m, str) else m[0]
        if w and w not in seen:
            seen.append(w)
    return seen


def mark_antecedent_distance(records: list[Record]) -> None:
    """Distance, in utterances, to the nearest preceding person expression.

    `ctx_dist = None` means no candidate anywhere in the preceding paragraph —
    either a generic use of 他 or a heuristic miss. Those items are reported
    separately rather than silently counted as context-resolvable.
    """
    from collections import defaultdict

    per_para: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        per_para[r.uuid].append(r)
    for group in per_para.values():
        group.sort(key=lambda r: r.seq)
        for i, r in enumerate(group):
            if not r.item_type:
                continue
            r.n_candidates = len(
                {c for prev in group[:i] for c in antecedent_candidates(prev.text)}
            )
            for back in range(1, i + 1):
                if PERSON.search(group[i - back].text):
                    r.ctx_dist = back
                    break


# --------------------------------------------------------------------- summary


def summarise(records: list[Record]) -> dict:
    c = Counter()
    c["sentences"] = len(records)
    c["with_context"] = sum(1 for r in records if r.ctx)
    for r in records:
        if r.item_type and not r.resolved_referent and r.candidates:
            c[f"item:{r.item_type}"] += 1
            if r.prior_correct is False:
                c[f"prior_adversarial:{r.item_type}"] += 1
        if r.resolved_referent:
            c[f"resolved:{r.item_type}"] += 1
        if r.agreeing_verb:
            c["agreeing_verb"] += 1
        if r.item_type == "anaphoric" and r.candidates:
            if r.ctx_dist is None:
                c["anaphoric:no_antecedent_found"] += 1
            else:
                for k in (1, 2, 4, 8):
                    if r.ctx_dist <= k:
                        c[f"anaphoric:antecedent_within_{k}"] += 1
            if r.n_candidates >= 2:
                c["anaphoric:ambiguous_2plus_candidates"] += 1
    return dict(c)
