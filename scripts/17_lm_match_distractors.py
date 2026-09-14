#!/usr/bin/env python3
"""Re-choose each lexical item's distractors so text alone cannot answer it.

    python3 scripts/17_lm_match_distractors.py --data data/lex --device cpu

`16_build_lexitems.py` selects distractors on lexical grounds — same POS,
frequency-matched, visually distinct sign. Those constraints say nothing about
whether the resulting sentence is *fluent*, and inspection of the first build
showed the gap plainly: a frequency-matched swap turned a verb of
cooking into a verb of seeing, and a human subject into an abstract
noun, in both cases leaving a sentence no speaker would produce.
A distractor that broken is discarded by any language model
without reference to the video, which would turn the diagnostic into a fluency
test — the one failure mode the whole design exists to avoid, and the one the
pronoun set avoided for free, since swapping 我 for 你 leaves a perfectly
grammatical sentence and only the referent changes.

So each item's frequency-matched pool is scored with a text-only LLM under the
same rule the real evaluation uses (mean per-token log-likelihood of the whole
candidate sentence, no video), and the three distractors kept are those whose
score sits **closest to gold's**. Items that cannot field three within `--tau`
nats of gold are dropped rather than reported: an item where every available
distractor is implausible is not a hard item, it is a fluency item.

Fluency is judged under *both* prompts, and a distractor must pass under both.
The first pass matched on the no-context prior alone and the gap showed up
immediately in `07_shortcuts.py`: a text-only LM answered 49.4% of the kept items
with no context and 79.7% once the preceding Chinese was supplied. Matching on
one prompt and evaluating under the other is not a hard item set, it is a hard
item set for a question nobody asks. `--ctx-k` therefore defaults to the window
`04_train.py` uses, and both score sets must agree that a distractor is viable.

Two things this is not. It is not tuning the items to beat the evaluated systems
— the matcher is Qwen, a decoder-only Chinese LM, while every evaluated system is
mT5-based, and the matching removes the text-prior component from all of them
equally rather than favouring any. And it is not a substitute for the shortcut
baselines: `07_shortcuts.py` must still be run afterwards, on the same `--ctx-k`,
because these scores choose the distractors and a baseline that shares their
arithmetic cannot be the thing that certifies them.

Writes `records.jsonl` back in place (after `records.prelm.jsonl` backup) so
`05_eval.py --data data/lex` needs no argument changes.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsl.conditions import PROMPT_CTX, PROMPT_PLAIN, last_k  # noqa: E402
from tsl.lexitems import text_is_clean  # noqa: E402

# The instruction the shortcut baselines use: same wording, minus the claim that
# a sign-language video is present, because none is.
_SIGN, _TEXT = "把這段台灣手語翻譯成中文：\n", "這句話的中文是：\n"
TEXT_PROMPT = PROMPT_PLAIN.replace(_SIGN, _TEXT)

N_KEEP = 3
# Separates prompt from candidate in a cache key. The prompt carries the context
# sentences, so keying on the candidate alone would collide across items whenever
# two of them happen to share a sentence — silently, and with the wrong score.
CACHE_SEP = "\n<|cand|>\n"


def prompt_for_item(r: dict, ctx_k: int) -> str:
    """The text-only prompt for this item, with the same window the systems get."""
    ctx = last_k(r["ctx"], ctx_k) if ctx_k and r.get("ctx") else []
    if not ctx:
        return TEXT_PROMPT
    return PROMPT_CTX.format(ctx=" ".join(c["text"] for c in ctx)).replace(_SIGN, _TEXT)


def score_batched(prompts: list[str], texts: list[str], llm: str, device: str,
                  threads: int, batch: int) -> list[float]:
    """Mean per-token log-likelihood of each text under `llm`, given its prompt.

    Batched, unlike `07_shortcuts.lm_scores`, which scores one candidate per
    forward pass. That is affordable for 975 pronoun items on a GPU and is not
    affordable for ~21k lexical candidates on a CPU, which is where this has to
    run — the card belongs to someone else tonight. The two agree: on the plain
    prompt both put gold top for 469 of the 949 items kept by the first pass.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "cpu":
        torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(llm)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(llm, dtype=dtype).to(device).eval()

    # One tokenisation per distinct prompt, not per row: with context there is one
    # prompt per item and ~41 candidates share it.
    pcache: dict[str, "torch.Tensor"] = {}
    for p in prompts:
        if p not in pcache:
            pcache[p] = tok(p, return_tensors="pt").input_ids[0]
    pids = [pcache[p] for p in prompts]
    enc = [tok(t, add_special_tokens=False, return_tensors="pt").input_ids[0]
           for t in texts]
    # Length-sorted batches: padding to the longest member is wasted compute, and
    # total lengths here span an order of magnitude. Sorted on prompt+candidate,
    # since with context the prompt is the larger and more variable half.
    order = sorted(range(len(enc)), key=lambda i: len(pids[i]) + len(enc[i]))
    out = [0.0] * len(enc)
    t0 = time.time()
    for b0 in range(0, len(order), batch):
        idx = order[b0:b0 + batch]
        rows = [torch.cat([pids[i], enc[i]]) for i in idx]
        m = max(len(r) for r in rows)
        ids = torch.full((len(rows), m), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(rows), m), dtype=torch.long)
        lab = torch.full((len(rows), m), -100, dtype=torch.long)
        for j, r in enumerate(rows):
            np_ = len(pids[idx[j]])
            ids[j, :len(r)] = r
            att[j, :len(r)] = 1
            lab[j, np_:len(r)] = r[np_:]
        ids, att, lab = ids.to(device), att.to(device), lab.to(device)
        with torch.no_grad():
            lg = model(input_ids=ids, attention_mask=att).logits.float()
        lp = torch.log_softmax(lg[:, :-1], -1)
        tgt = lab[:, 1:]
        keep = tgt != -100
        got = lp.gather(-1, tgt.clamp(min=0)[..., None])[..., 0] * keep
        n = keep.sum(1).clamp(min=1)
        for j, i in enumerate(idx):
            out[i] = float(got[j].sum() / n[j])
        done = b0 + len(idx)
        if b0 % (batch * 20) == 0 or done == len(order):
            el = time.time() - t0
            print(f"  {done}/{len(order)}  {el/60:.1f} min elapsed, "
                  f"~{el/max(1,done)*(len(order)-done)/60:.0f} min left", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/lex"))
    ap.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.35,
                    help="max |mean logprob - gold| for a distractor to be usable")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the yield/tau curve without rewriting records")
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N items (smoke test; implies "
                         "--dry-run, since a partial pass must never be written)")
    ap.add_argument("--source", default=None,
                    help="records to select from (default records.jsonl). Pass "
                         "records.prelm.jsonl to re-select from the pristine "
                         "build after an earlier pass has already narrowed it.")
    ap.add_argument("--ctx-k", type=int, default=4,
                    help="context sentences in the matching prompt; must match "
                         "04_train.py --ctx-k (default 4). 0 = no context")
    ap.add_argument("--cache", type=Path, default=None,
                    help="path for the per-candidate score cache (default "
                         "<data>/lm_scores.json, or lm_scores_k<N>.json with "
                         "context — the two must never share a file)")
    ap.add_argument("--plain-cache", type=Path, default=None,
                    help="scores under the no-context prompt (default "
                         "<data>/lm_scores.json). Needed whenever --ctx-k > 0, "
                         "so the pass refuses to run without them rather than "
                         "quietly reporting half the picture.")
    ap.add_argument("--match-on", choices=["plain", "ctx", "both"], default="plain",
                    help="which prompt the tau window and the rank balance are "
                         "computed under. `plain` is the default because the "
                         "conditions this item set is built for -- local and "
                         "blank_plain -- are in neither CTX_ set and so are "
                         "scored under PROMPT_PLAIN. Matching under a prompt no "
                         "reported condition uses would control the wrong leak.")
    ap.add_argument("--no-balance", action="store_true",
                    help="keep the three distractors nearest gold instead of "
                         "balancing gold's rank. Nearest-three bounds how much "
                         "gold wins by and not whether it wins: it left a "
                         "text-only LM at 49.4%%, and tightening tau does not "
                         "help because the survivors are still mostly below gold.")
    ap.add_argument("--max-use", type=int, default=0,
                    help="cap on how many items one word may serve as a "
                         "distractor for (0 = uncapped). Matching on frequency "
                         "pulls every item toward the same few frequent words; "
                         "the cap spreads them back out.")
    ap.add_argument("--from-cache", action="store_true",
                    help="re-select from a previous pass's cache without "
                         "re-scoring. Scoring is ~3.3 h on CPU and tau is a "
                         "judgement call, so it must be re-tunable in seconds.")
    a = ap.parse_args()
    if a.limit:
        a.dry_run = True
    cache_path = a.cache or (a.data / (f"lm_scores_k{a.ctx_k}.json"
                                       if a.ctx_k else "lm_scores.json"))
    plain_path = a.plain_cache or (a.data / "lm_scores.json")
    if a.ctx_k and not plain_path.exists():
        sys.exit(f"--ctx-k {a.ctx_k} needs the no-context scores too, and "
                 f"{plain_path} is missing. Run with --ctx-k 0 first.")

    path = a.data / (a.source or "records.jsonl")
    recs = [json.loads(l) for l in path.open()]
    items = [r for r in recs if r.get("item_type") == "lexical"]
    if a.limit:
        items = items[:a.limit]
    print(f"selecting from {path}")
    print(f"{len(items)} lexical items; scoring gold + pool with {a.llm} "
          f"on {a.device}")

    # One flat list so the batcher can sort by length across the whole job.
    texts: list[str] = []
    prompts: list[str] = []
    spans: list[tuple[int, int, list[str]]] = []
    for r in items:
        gold_w = r["gold_word"]
        pool = r.get("distractor_pool") or r["distractors"]
        start = len(texts)
        texts.append(r["candidates"][0])
        for w in pool:
            texts.append(r["candidates"][0].replace(gold_w, w, 1))
        # Every candidate of an item is scored under that item's own context, so
        # the comparison the tau threshold makes is within-item throughout.
        prompts += [prompt_for_item(r, a.ctx_k)] * (len(texts) - start)
        spans.append((start, len(texts), list(pool)))
    n_ctx = sum(1 for p in prompts if p != TEXT_PROMPT)
    print(f"{len(texts)} sequences to score "
          f"(ctx_k={a.ctx_k}; {n_ctx} carry context, {len(texts) - n_ctx} plain)")

    # Keyed by prompt+candidate, not by position: a later build with a different
    # pool still hits for every sentence it shares under the same context.
    keys = [p + CACHE_SEP + t if a.ctx_k else t for p, t in zip(prompts, texts)]
    if a.from_cache:
        cached = json.loads(cache_path.read_text())
        sc = [cached[k] for k in keys]    # KeyError here means the cache is for
        print(f"loaded {len(sc)} scores from {cache_path}")  # a different build
    else:
        sc = score_batched(prompts, texts, a.llm, a.device, a.threads, a.batch)
        cache_path.write_text(json.dumps(dict(zip(keys, sc)), ensure_ascii=False))
        print(f"cached {len(texts)} scores to {cache_path}")

    # The no-context scores, which every distractor must satisfy as well.
    if a.ctx_k:
        plain = json.loads(plain_path.read_text())
        missing = [t for t in texts if t not in plain]
        if missing:
            sys.exit(f"{len(missing)} candidates are absent from {plain_path} "
                     f"(first: {missing[0]!r}). That cache was built from a "
                     f"different pool; rebuild it with --ctx-k 0 --source "
                     f"{a.source or 'records.jsonl'} before matching on context.")
        psc = [plain[t] for t in texts]
    else:
        psc = sc

    kept, dropped, broken = [], 0, 0
    curve: dict[float, int] = {t: 0 for t in (0.15, 0.25, 0.35, 0.5, 0.75, 1.0)}
    viable: list[dict] = []
    for r, (s, e, pool) in zip(items, spans):
        gc = [abs(sc[s + 1 + i] - sc[s]) for i in range(len(pool))]
        gp = [abs(psc[s + 1 + i] - psc[s]) for i in range(len(pool))]
        # Under `both` a distractor is only as good as its worse prompt, which is
        # what makes tau one joint constraint rather than two a word could satisfy
        # alternately.
        g = {"ctx": gc, "plain": gp,
             "both": [max(x, y) for x, y in zip(gc, gp)]}[a.match_on]
        gaps = sorted(zip(g, pool))
        for t in curve:
            if len(gaps) >= N_KEEP and gaps[N_KEEP - 1][0] <= t:
                curve[t] += 1
        bad = not text_is_clean(r["candidates"][0])
        if bad or len(gaps) < N_KEEP or gaps[N_KEEP - 1][0] > a.tau:
            r["item_type"] = None   # no fluent distractor, or damaged source text
            r["candidates"] = []
            dropped += not bad
            broken += bad
            continue
        # Split the in-tau words by which side of gold they land on. Under
        # `both` a word must agree with itself across the two prompts to count
        # for either side: one that outscores gold with context and not without
        # cannot be placed at a known rank, so it is set aside.
        at = {w: s + 1 + i for i, w in enumerate(pool)}
        hi_w, lo_w = [], []
        for gv, w in gaps:
            if gv > a.tau:
                break
            up_c, up_p = sc[at[w]] > sc[s], psc[at[w]] > psc[s]
            up = {"ctx": up_c, "plain": up_p, "both": up_c and up_p}[a.match_on]
            dn = {"ctx": not up_c, "plain": not up_p,
                  "both": not up_c and not up_p}[a.match_on]
            if up:
                hi_w.append(w)
            elif dn:
                lo_w.append(w)
        v = {"r": r, "s": s, "at": at, "gap": dict(zip(pool, g)), "rank": None,
             "hi": hi_w, "lo": lo_w, "fluent": [w for gv, w in gaps if gv <= a.tau],
             "lo_rank": max(1, 4 - len(lo_w)), "hi_rank": min(4, len(hi_w) + 1)}
        if v["lo_rank"] > v["hi_rank"]:
            r["item_type"] = None   # in tau, but the two prompts never agree
            r["candidates"] = []
            dropped += 1
            continue
        viable.append(v)

    # Give gold a target rank per item so that its rank is uniform over the kept
    # set. A text-only model then answers at chance *by construction* instead of
    # at whatever tau happens to buy — tau bounds how much gold wins by, not
    # whether it wins, which is why the first pass still read 49.4%.
    def assign(q: int) -> dict[int, int] | None:
        out: dict[int, int] = {}
        taken: set[int] = set()
        # Scarcest rank first, least flexible item first. Rank 4 needs three
        # distractors that all outscore gold and only a minority of items can
        # field them; spending those items on rank 1 would strand the quota.
        for rank in (4, 3, 2, 1):
            cand = [j for j, v in enumerate(viable)
                    if j not in taken and v["lo_rank"] <= rank <= v["hi_rank"]]
            cand.sort(key=lambda j: (viable[j]["hi_rank"] - viable[j]["lo_rank"], j))
            if len(cand) < q:
                return None
            for j in cand[:q]:
                out[j] = rank
                taken.add(j)
        return out

    if not a.no_balance:
        best, lo, hi = {}, 1, len(viable) // 4
        while lo <= hi:            # the largest quota every rank can fill
            mid = (lo + hi) // 2
            got = assign(mid)
            if got:
                best, lo = got, mid + 1
            else:
                hi = mid - 1
        for j, v in enumerate(viable):
            v["rank"] = best.get(j)
            if v["rank"] is None:
                v["r"]["item_type"] = None   # balancing has no slot for it
                v["r"]["candidates"] = []
                dropped += 1
        viable = [v for v in viable if v["rank"] is not None]

    # Most-constrained-first. The cap below is a budget, and processing in file
    # order would spend it on whichever items happen to come early, leaving the
    # items with the fewest fluent options to fall back and defeat the cap.
    used: Counter[str] = Counter()
    argmax = Counter()
    ranks: Counter[int] = Counter()
    for v in sorted(viable, key=lambda v: len(v["fluent"])):
        r, at, s, rank = v["r"], v["at"], v["s"], v["rank"]
        gold, pgold, gap_of = sc[s], psc[s], v["gap"]
        freq = dict(zip(r.get("distractor_pool") or [], r.get("pool_freq") or []))
        lg = math.log(max(1, r["gold_freq"]))

        def pick(words: list[str], n: int) -> list[str]:
            """The `n` frequency-closest of `words`.

            Fluency is a threshold, frequency is the tie-break. Every word here is
            already inside tau and on the side of gold its slot requires, so the
            freedom that remains is spent matching gold's corpus frequency; the
            first pass showed what ignoring it costs, with the marginal-frequency
            guesser answering 40.3% of the kept items.
            """
            room = [w for w in words if used[w] < a.max_use] if a.max_use else words
            if len(room) < n:
                room = words   # the cap yields rather than costing an item
            return sorted(room, key=lambda w: (
                abs(math.log(max(1, freq.get(w, 1))) - lg), w))[:n]

        if rank is None:
            alts = sorted(pick(v["fluent"], N_KEEP))
        else:
            # rank r means gold is the r-th likeliest of the four, so exactly
            # r-1 distractors must outscore it.
            alts = sorted(pick(v["hi"], rank - 1) + pick(v["lo"], 4 - rank))
            ranks[rank] += 1
        used.update(alts)
        r["distractors"] = alts
        r["candidates"] = [r["candidates"][0]] + [
            r["candidates"][0].replace(r["gold_word"], w, 1) for w in alts]
        r["lm_gold"] = round(gold, 4)
        r["lm_gap"] = round(max(gap_of[w] for w in alts), 4)
        r["lm_ctx_k"] = a.ctx_k
        r["lm_match_on"] = a.match_on
        r["lm_rank"] = rank
        # Gold-is-argmax under each prompt is exactly what 07_shortcuts.py reports
        # as lm_prior and lm_context, so the leak both thresholds exist to close is
        # readable here without a second hour of scoring. It does not replace that
        # run: these are the scores that chose the distractors.
        argmax["ctx"] += all(sc[at[w]] < gold for w in alts)
        argmax["plain"] += all(psc[at[w]] < pgold for w in alts)
        # The build-time `prior_correct` described a different trio. Left stale it
        # would misreport the frequency prior on exactly the items the paper
        # reports, so it is recomputed against the distractors actually kept.
        pool = [(r["gold_freq"], r["gold_word"])] + [(freq.get(w, 0), w) for w in alts]
        r["prior_correct"] = max(pool)[1] == r["gold_word"]
        kept.append(r)

    print(f"\nkept {len(kept)}, dropped {dropped} (tau={a.tau}), "
          f"{broken} with damaged source text")
    print("  yield by tau: " + "  ".join(
        f"{t}: {n}" for t, n in sorted(curve.items())))
    if kept:
        med = sorted(r["lm_gap"] for r in kept)[len(kept) // 2]
        print(f"  median gap of the worst kept distractor: {med:.3f} nats")
        pr = sum(1 for r in kept if r["prior_correct"]) / len(kept)
        print(f"  frequency prior answers {pr:.1%} of kept items  (chance 25%)")
        print(f"  gold is argmax for the text-only LM: "
              f"{argmax['plain'] / len(kept):.1%} plain, "
              f"{argmax['ctx'] / len(kept):.1%} with ctx_k={a.ctx_k}  (chance 25%)")
        if ranks:
            print("  gold's rank among the four, by construction: " + "  ".join(
                f"{k}:{n} ({n / len(kept):.0%})" for k, n in sorted(ranks.items())))
        slots = Counter(w for r in kept for w in r["distractors"])
        top10 = sum(n for _, n in slots.most_common(10)) / max(1, sum(slots.values()))
        print(f"  distractor types {len(slots)}; commonest fills "
              f"{slots.most_common(1)[0][1]} slots, top-10 take {top10:.0%}")

    if a.dry_run:
        print("dry run: records.jsonl not rewritten")
        return
    # Always write records.jsonl, never `path`: with --source records.prelm.jsonl
    # the read path *is* the pristine backup, and writing back to it would
    # destroy the only copy of the unnarrowed build.
    out_path = a.data / "records.jsonl"
    backup = a.data / "records.prelm.jsonl"
    if not backup.exists():
        shutil.copy2(out_path, backup)
        print(f"  backed up original to {backup}")
    with out_path.open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"rewrote {out_path}")


if __name__ == "__main__":
    main()
