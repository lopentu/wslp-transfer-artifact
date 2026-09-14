"""The experimental conditions, and everything about them that is pure data.

Kept free of torch so the condition logic — which is where a subtle mistake would
quietly invalidate the paper — can be unit-tested without a GPU environment.

Every condition in plan §6.1/§6.2 is a *view* of the same record, so one trained
model is evaluated under all of them and no difference between conditions can be a
difference between models:

| condition        | target video | context video        | context text |
|------------------|--------------|----------------------|--------------|
| local            | yes          | -                    | -            |
| video            | yes          | correct, preceding   | -            |
| text             | yes          | -                    | correct      |
| both             | yes          | correct, preceding   | correct      |
| mismatch_video   | yes          | other paragraph      | -            |
| mismatch_text    | yes          | -                    | other para   |
| mismatch_both    | yes          | other paragraph      | other para   |
| shuffle_video    | yes          | same para, not prev  | -            |
| blank_video      | zeroed       | correct, preceding   | correct      |
| text_only        | zeroed       | -                    | correct      |
| blank_plain      | zeroed       | -                    | -            |
| swap_plain       | ANOTHER clip | -                    | -            |
| shuffle_frames   | frames shuffled | -                 | -            |

`swap_plain` and `shuffle_frames` were added on 2026-08-18, and the reason is an
objection to `blank_plain` rather than to anything it measures. A zeroed pose
sequence is not merely uninformative, it is *out of distribution*: no clip in
training ever had 133 keypoints pinned at the origin, so `local` - `blank_plain`
confounds "how much the target clip is worth" with "how the network reacts to a
degenerate input". That confound is load-bearing exactly where the negative
deltas are, so it needs its own control rather than a caveat.

Both new conditions keep the input in distribution and break only the
correspondence between clip and sentence:

    swap_plain      the target clip of a DIFFERENT utterance, from a different
                    paragraph, matched for duration. Real pose, real motion, real
                    signer -- wrong sentence. A model that reads clip-specific
                    lexical information must lose accuracy here; a model running a
                    video-independent language prior must not.
    shuffle_frames  this clip's own frames in a permuted order. Identical frame
                    distribution to `local` down to the single frame, so what it
                    removes is temporal structure alone.

Both are in neither CTX_ set and both keep PROMPT_PLAIN, so each differs from
`local` in exactly one factor, as `blank_plain` does.

`blank_plain` is the only condition that removes the video and changes *nothing
else*. It was added on 2026-08-14, after an audit found that the paper was
reading "what the target video is worth" off `text_only` — which zeroes the video
but simultaneously supplies the context text and, through `prompt_for`, swaps
PROMPT_PLAIN for PROMPT_CTX. Since every reported cell trains on `local` only,
`text_only` is doubly out of distribution, and a drop from `local` to it confounds
three changes. `blank_plain` differs from `local` in exactly one.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

CONDITIONS = (
    "local", "video", "text", "both",
    "mismatch_video", "mismatch_text", "mismatch_both",
    "shuffle_video", "blank_video", "text_only", "blank_plain",
    "swap_plain", "shuffle_frames", "swap_within",
)

CTX_VIDEO = {"video", "both", "blank_video"}
CTX_TEXT = {"text", "both", "blank_video", "text_only"}
MISMATCH_VIDEO = {"mismatch_video", "mismatch_both"}
MISMATCH_TEXT = {"mismatch_text", "mismatch_both"}
# `blank_plain` is deliberately in this set and in *neither* CTX_ set: zeroed
# target, no context of either kind, PROMPT_PLAIN. That makes local -> blank_plain
# a one-factor ablation; local -> text_only does not.
BLANK_TARGET = {"blank_video", "text_only", "blank_plain"}
# The in-distribution target ablations. Deliberately in neither CTX_ set and
# not in BLANK_TARGET: the target clip is replaced or reordered, never zeroed.
#
# `swap_within` was added 2026-08-22 to answer a reviewer's objection to
# `swap_plain` rather than to anything it measures. `swap_plain` draws its donor
# from a DIFFERENT paragraph, which was a deliberate choice -- a same-paragraph
# donor shares topic and lexis, so a model could score above chance on it without
# reading the clip -- but it changes the signer, the recording session and the
# discourse topic along with the clip-sentence correspondence. So part of the
# 22.1-point drop could be signer or session mismatch. `swap_within` breaks only
# the correspondence: same paragraph, therefore same signer and same session, a
# different utterance, duration-matched. It is the conservative bound on the
# effect and `swap_plain` the liberal one, and reporting both is what makes the
# number clean without using signer metadata as a variable (Ethics).
SWAP_TARGET = {"swap_plain", "swap_within"}
SWAP_WITHIN = {"swap_within"}
SHUFFLE_FRAMES = {"shuffle_frames"}

PROMPT_PLAIN = "把這段台灣手語翻譯成中文：\n"
PROMPT_CTX = "前文：{ctx}\n把這段台灣手語翻譯成中文：\n"


def assign_donors(records: list[dict], seed: int = 0) -> dict[str, str]:
    """Pair each record with a mismatched-context donor.

    Same paragraph type, different paragraph, same theme where possible, closest
    duration. Matching on those three keeps the mismatched condition a test of
    *referential* compatibility rather than a test of whether the context looks
    superficially out of place — an obviously alien context would be rejected for
    reasons that have nothing to do with reference.
    """
    rng = random.Random(seed)
    pool = [r for r in records if r["ctx"]]
    out: dict[str, str] = {}
    by_theme: dict[tuple[str, str], list[dict]] = {}
    by_type: dict[str, list[dict]] = {}
    for r in pool:
        by_theme.setdefault((r["para_type"], r["theme"]), []).append(r)
        by_type.setdefault(r["para_type"], []).append(r)

    for r in pool:
        dur = r["t2"] - r["t1"]
        for bucket in (by_theme.get((r["para_type"], r["theme"]), []),
                       by_type.get(r["para_type"], [])):
            cands = [c for c in bucket if c["uuid"] != r["uuid"]]
            if not cands:
                continue
            cands.sort(key=lambda c: (abs((c["t2"] - c["t1"]) - dur), c["key"]))
            # Draw from the closest few rather than the single closest, so donors
            # are not all concentrated on a handful of median-length utterances.
            out[r["key"]] = rng.choice(cands[: max(3, len(cands) // 20)])["key"]
            break
    return out


def assign_target_swaps(records: list[dict], seed: int = 0, perm: int = 0,
                        pool: list[dict] | None = None,
                        window: int = 8) -> dict[str, str]:
    """Pair each record with a donor for its own TARGET clip (`swap_plain`).

    Three constraints, each answering a way the control could be read as
    something other than "the right kind of clip, of the wrong sentence":

      different paragraph  a donor from the same paragraph shares topic, signer
                           and often lexis, so a model could score above chance on
                           it for reasons that are not clip-specific.
      matched duration     the closest `window` candidates by duration, drawn from
                           at random. Sequence length is the one property of the
                           input a model could exploit without reading anything,
                           and it is also what `blank_plain` leaves untouched.
      donors from `pool`   the split the caller passes, so an evaluation swaps in
                           clips that are held out exactly as the target is.

    `perm` selects an independent draw over the same candidate windows: the
    reported figure is the mean over several, since any single assignment could be
    lucky about which wrong clip a given item receives.
    """
    rng = random.Random(1_000_003 * seed + 7_919 * perm + 13)
    src = list(records)
    dst = list(pool if pool is not None else records)
    by_uuid: dict[str, list[dict]] = {}
    for c in dst:
        by_uuid.setdefault(c["uuid"], []).append(c)
    out: dict[str, str] = {}
    for r in src:
        dur = r["t2"] - r["t1"]
        cands = [c for c in dst if c["uuid"] != r["uuid"]]
        if not cands:
            continue
        cands.sort(key=lambda c: (abs((c["t2"] - c["t1"]) - dur), c["key"]))
        out[r["key"]] = rng.choice(cands[:max(1, min(window, len(cands)))])["key"]
    return out


def assign_within_swaps(records: list[dict], seed: int = 0, perm: int = 0,
                        pool: list[dict] | None = None,
                        window: int = 2) -> dict[str, str]:
    """`swap_within`: a donor clip from the target's OWN paragraph.

    Same two constraints as `assign_target_swaps` except the first, which is
    inverted: the donor must share the target's `uuid` rather than differ from it.
    Sharing the paragraph holds the signer, the recording session and the topic
    fixed, so what is left varying is the correspondence between this clip and
    this sentence.

    The cost is that a same-paragraph donor also shares lexis, so a model reading
    only paragraph-level information could still score. That is exactly why this
    is reported *beside* `swap_plain` and not instead of it: the two bracket the
    effect, and a drop that survives the within-paragraph version cannot be
    attributed to signer or session change.

    `window` is 2 rather than `assign_target_swaps`'s 8, and the reason is the
    reduced pool: a test paragraph holds a median 12 utterances, so the closest 8
    of them are nearly all of them and the duration match degrades to a median
    940 ms. At 2 it is 280 ms, against 10 ms when the whole split is available.
    That gap is a real cost of the control and belongs in the appendix beside the
    number: holding the signer fixed and matching duration tightly cannot both be
    done inside one paragraph, so `swap_plain` and `swap_within` each give up one.

    Records whose paragraph holds no other utterance are absent from the mapping,
    and `TSLDataset` leaves those items at their own clip -- so the condition is
    scored on a subset, which the caller must report. `n_swapped` in the eval row
    count is how that is checked.
    """
    rng = random.Random(2_000_003 * seed + 7_919 * perm + 29)
    dst = list(pool if pool is not None else records)
    by_uuid: dict[str, list[dict]] = {}
    for c in dst:
        by_uuid.setdefault(c["uuid"], []).append(c)
    out: dict[str, str] = {}
    for r in records:
        dur = r["t2"] - r["t1"]
        cands = [c for c in by_uuid.get(r["uuid"], []) if c["key"] != r["key"]]
        if not cands:
            continue
        cands.sort(key=lambda c: (abs((c["t2"] - c["t1"]) - dur), c["key"]))
        out[r["key"]] = rng.choice(cands[:max(1, min(window, len(cands)))])["key"]
    return out


def last_k(ctx: list[dict], k: int | None) -> list[dict]:
    """The k utterances immediately preceding the target (records store more).

    k=None means every stored utterance. Records carry up to 8; the nearest
    candidate antecedent of an anaphoric item is a median 2 utterances back, but
    a 2-utterance window reaches it only 58% of the time against 75% at 4 and 94%
    for the whole paragraph — so k is an experimental variable, not a constant.
    """
    if k is None or k >= len(ctx):
        return ctx
    return ctx[-k:] if k > 0 else []


def context_text(rec: dict, cond: str, donors: dict[str, str],
                 by_key: dict[str, dict], k: int | None = None) -> str | None:
    if cond in MISMATCH_TEXT:
        donor = by_key.get(donors.get(rec["key"], ""))
        if donor is None or not donor["ctx"]:
            return None
        return " ".join(c["text"] for c in last_k(donor["ctx"], k))
    if cond in CTX_TEXT and rec["ctx"]:
        return " ".join(c["text"] for c in last_k(rec["ctx"], k))
    return None


def prompt_for(ctx_txt: str | None) -> str:
    return PROMPT_CTX.format(ctx=ctx_txt) if ctx_txt else PROMPT_PLAIN


def load_records(path: Path | str) -> list[dict]:
    return [json.loads(l) for l in Path(path).open()]


def split_records(records: list[dict], folds: dict, fold: int) -> dict[str, list[dict]]:
    f = folds["folds"][fold]
    where: dict[str, str] = {}
    for name in ("train", "dev", "test"):
        for u in f[name]:
            where[u] = name
    out: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for r in records:
        s = where.get(r["uuid"])
        if s:
            out[s].append(r)
    return out
