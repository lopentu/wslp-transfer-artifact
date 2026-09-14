#!/usr/bin/env python3
"""Shortcut baselines — run this BEFORE training. It is a go/no-go gate.

Every one of these solves the contrastive task without looking at the video:

  majority        always pick the candidate whose pronoun is commonest in train
  lm_prior        base LLM scores the candidates, no visual tokens, no context
  lm_context      base LLM scores the candidates given the context TEXT only

If `lm_context` already scores near-ceiling on the anaphoric items, the items are
answerable from Chinese alone and the contrastive design has to change (harder
distractors, or drop the items where the preceding Chinese gives the pronoun away)
before a single GPU-hour goes into training. This is the plan §8 channel-only
baseline, and it is the objection a reviewer will raise first.

The gap `lm_context` -> trained model is also the honest measure of what the sign
video contributes, which no BLEU table can show.

    python3 scripts/07_shortcuts.py --llm Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsl.conditions import load_records, split_records  # noqa: E402
from tsl.metrics import cell, contrastive_accuracy  # noqa: E402
from tsl.splits import load as load_folds  # noqa: E402


def majority(items: list[dict], train: list[dict]) -> list[bool]:
    """Pick by the training-set frequency of the word each candidate contributes.

    Two item sets, one baseline. For pronoun items the contested word is the
    pronoun gloss and the alternatives are fixed by `ALTERNATES`; for lexical
    items it is `gold_word` against that item's own chosen distractors. Both
    count frequency over the *training* fold only, which is what makes this a
    baseline a system could actually have used rather than an oracle.
    """
    from tsl.items import ALTERNATES

    lex = Counter()
    pron = Counter()
    for r in train:
        if r.get("pron_gloss"):
            pron[r["pron_gloss"]] += 1
        if r.get("gold_word"):
            lex[r["gold_word"]] += 1

    hits = []
    for r in items:
        if r.get("item_type") == "lexical":
            gold, alts, freq = r["gold_word"], r["distractors"], lex
        else:
            gold, alts, freq = r["pron_gloss"], ALTERNATES.get(r["pron_gloss"], []), pron
        # Ties go to the alphabetically first word, not to gold: `max` would
        # otherwise hand the baseline every item whose candidates are all unseen
        # in train, and report a shortcut that is really a quirk of argmax.
        best = max(sorted([gold] + list(alts)), key=lambda w: freq.get(w, 0))
        hits.append(best == gold)
    return hits


def lm_scores(items: list[dict], llm: str, use_context: bool, batch: int = 16,
              device: str = "cuda", threads: int = 8, ctx_k: int | None = 4):
    """Score the candidates with a text-only LLM.

    `ctx_k` must be the window the evaluated systems get (`04_train.py --ctx-k`,
    4 everywhere so far). This baseline sits in the same table as those systems,
    so feeding it the whole context while they see the last four sentences is not
    a conservative choice, it is a different experiment: 65% of pronoun items and
    59% of lexical items carry more than four sentences of context.

    `device='cpu'` exists because this baseline is small enough to run beside a
    training sweep instead of contending with it for the shared card — and it is
    fp32 there, since CPU bfloat16 support is uneven. Threads are capped for the
    same reason the ONNX ones are (README §5): this box has other people on it.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tsl.conditions import PROMPT_CTX, PROMPT_PLAIN, last_k

    if device == "cpu":
        torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(llm)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(llm, dtype=dtype).to(device).eval()

    out: list[list[dict]] = []
    for r in items:
        ctx = last_k(r["ctx"], ctx_k) if use_context and r["ctx"] else []
        if ctx:
            prompt = PROMPT_CTX.format(ctx=" ".join(c["text"] for c in ctx))
        else:
            prompt = PROMPT_PLAIN
        # No visual tokens at all — the point is what text alone can do.
        prompt = prompt.replace("把這段台灣手語翻譯成中文：\n", "這句話的中文是：\n")
        per = []
        for cand in r["candidates"]:
            p = tok(prompt, return_tensors="pt").input_ids
            c = tok(cand, add_special_tokens=False, return_tensors="pt").input_ids
            ids = torch.cat([p, c], 1).to(device)
            labels = ids.clone()
            labels[:, : p.shape[1]] = -100
            with torch.no_grad():
                lg = model(input_ids=ids).logits.float()
            lp = torch.log_softmax(lg[:, :-1], -1)
            tgt = labels[:, 1:]
            keep = tgt != -100
            got = lp.gather(-1, tgt.clamp(min=0)[..., None])[..., 0] * keep
            n = int(keep.sum())
            per.append({"sum": float(got.sum()), "mean": float(got.sum() / max(1, n))})
        out.append(per)
    return out


def report(name: str, items: list[dict], hits: list[bool]) -> None:
    # Driven by what is in the file, not by a fixed list: the lexical set carries
    # item_type "lexical" and a hardcoded anaphoric/deictic pair would print
    # nothing at all for it, which reads as a clean run rather than a silent one.
    for t in sorted({r["item_type"] for r in items}):
        h = [x for x, r in zip(hits, items) if r["item_type"] == t]
        if h:
            print(f"  {name:24s} {t:10s} {cell(h).fmt()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--skip-lm", action="store_true", help="majority only, no GPU")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                    help="cpu runs beside a training sweep instead of contending "
                         "with it for the shared card; slower, same numbers")
    ap.add_argument("--threads", type=int, default=8, help="--device cpu only")
    ap.add_argument("--ctx-k", type=int, default=4,
                    help="context sentences the baseline sees; must match "
                         "04_train.py --ctx-k (default 4). 0 = no context")
    ap.add_argument("--out", type=Path, default=Path("data/shortcuts.json"))
    a = ap.parse_args()

    recs = load_records(a.data / "records.jsonl")
    folds = load_folds(a.data / "folds.json")

    # Pool the held-out items across folds, exactly as 06_report.py does, so these
    # numbers sit in the same table as the trained systems'.
    items: list[dict] = []
    train_by_fold: dict[str, list[dict]] = {}
    for i in range(folds["k"]):
        sp = split_records(recs, folds, i)
        for r in sp["test"]:
            if r["item_type"] and r["candidates"]:
                items.append(r)
                train_by_fold[r["key"]] = sp["train"]
    kinds = Counter(r["item_type"] for r in items)
    print(f"{len(items)} pooled held-out items "
          f"({', '.join(f'{n} {t}' for t, n in sorted(kinds.items()))})\n")
    print("chance = 25.0% (4 candidates)\n")

    results: dict[str, dict] = {}

    hits = [majority([r], train_by_fold[r["key"]])[0] for r in items]
    report("majority (train freq)", items, hits)
    results["majority"] = {r["key"]: h for r, h in zip(items, hits)}

    if not a.skip_lm:
        for name, use_ctx in (("lm_prior", False), ("lm_context", True)):
            sc = lm_scores(items, a.llm, use_ctx, device=a.device,
                           threads=a.threads, ctx_k=a.ctx_k)
            hits = contrastive_accuracy(sc, "mean")
            report(name, items, hits)
            results[name] = {r["key"]: h for r, h in zip(items, hits)}
            # Both normalisations, because the lexical distractors are rank-
            # balanced under mean and that guarantee does not carry over: an
            # unnormalised sum rewards short candidates, and the balance is a
            # property of (item set x scoring rule). Reporting only the rule the
            # set was built for would be assuming the thing a reviewer checks.
            hits_s = contrastive_accuracy(sc, "sum")
            report(name + " [sum]", items, hits_s)
            results[name + "_sum"] = {r["key"]: h for r, h in zip(items, hits_s)}

    # `device` is recorded because it selects the dtype (fp32 on CPU, bfloat16 on
    # CUDA) and the two do not agree to the decimal: the lexical lm_prior is
    # 25.0% in fp32 and 25.7% in bf16 on the same items. Neither is wrong, but a
    # later re-run that lands on the other one should be able to tell why.
    a.out.write_text(json.dumps(
        {"llm": a.llm, "n_items": len(items), "ctx_k": a.ctx_k,
         "device": a.device, "dtype": "float32" if a.device == "cpu" else "bfloat16",
         "items": [{"key": r["key"], "type": r["item_type"],
                    "gloss": r["pron_gloss"] or r.get("gold_word")}
                   for r in items],
         "hits": results}, ensure_ascii=False))
    print(f"\n-> {a.out}")
    print("\nGATE: if lm_context on anaphoric items is close to a trained model, the")
    print("items are answerable from Chinese alone. Tighten the distractors before")
    print("spending GPU time — see README 'Risks'.")


if __name__ == "__main__":
    main()
