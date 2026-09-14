#!/usr/bin/env python3
"""Evaluate one checkpoint under every condition, and dump per-item results.

Contrastive scoring is the main pass and is cheap — one forward per candidate, no
decoding. Free generation is run only when asked, and only for the conditions the
paper actually reports translations for.

    python3 scripts/05_eval.py --ckpt runs/f0_context_csl_daily_lora_k4_s0/best.pt
    python3 scripts/05_eval.py --ckpt ... --generate --conditions local,both

The checkpoint says which model to rebuild, so this script does not need to be
told. A Uni-Sign run under LoRA does not carry its frozen mT5, so `tsl.unisign`
reloads the initialisation checkpoint the run recorded and puts the trained
tensors on top — see `unisign.load_run`.

Scoring runs in fp32 by default. Contrastive accuracy is an argmax over
candidate log-likelihoods whose margins can be a fraction of a nat, and bf16
carries about three decimal digits: `--amp bf16` is faster and available, but the
reported numbers should come from the default.

Writes `<run>/eval.json` (per-item rows) — 06_report.py turns those into tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsl import unisign as us  # noqa: E402
from tsl.conditions import (  # noqa: E402
    CONDITIONS, SWAP_TARGET, SWAP_WITHIN, assign_donors, assign_target_swaps,
    assign_within_swaps, load_records, split_records,
)
from tsl.dataset import TSLDataset, collate  # noqa: E402
from tsl.items import ALTERNATES  # noqa: E402
from tsl.metrics import (  # noqa: E402
    choice, contrastive_accuracy, margin, pronoun_slot_hit,
)
from tsl.splits import load as load_folds  # noqa: E402

DEFAULT_CONDITIONS = (
    "local", "video", "text", "both",
    "mismatch_video", "mismatch_text", "mismatch_both",
    "shuffle_video", "text_only",
)


# --------------------------------------------------------------------- systems


def to_cuda(batch):
    if torch.is_tensor(batch):
        return batch.cuda()
    if isinstance(batch, dict):
        return {k: to_cuda(v) for k, v in batch.items() if k != "meta"}
    return batch


#: Both names of the output projection in a `UniSignTSL` state dict. `self.mt5`
#: is `self.core.mt5_model`, so the two keys alias one tensor and writing the
#: module's parameter updates both.
HEAD_KEYS = ("core.mt5_model.lm_head.weight", "mt5.lm_head.weight")


def graft_head(model, donor_ckpt: Path) -> dict:
    """Replace a *fine-tuned* model's `lm_head` with another run's fine-tuned one.

    The paper's §3.4 swap happens at initialization, so it cannot separate the two
    readings its own Discussion leaves open: whether a mismatched output
    projection stops the network from learning to use the video, or whether the
    video is used and the readout cannot express it. All three reviews name the
    same control -- swap the head *after* fine-tuning, no retraining -- and this
    is it. A rescue implicates the readout; no rescue implicates the trajectory.

    Deliberately not a `load_state_dict`: the donor and recipient differ in every
    other tensor, and a strict=False load of a 911-key state dict would silently
    move whatever else happened to match. One tensor moves, by name, or this
    raises.
    """
    dck = torch.load(donor_ckpt, map_location="cpu", weights_only=False, mmap=True)
    if dck.get("arch") != "unisign":
        raise SystemExit(f"{donor_ckpt} is arch={dck.get('arch')!r}, not unisign")
    src = dck["state"]
    have = [k for k in HEAD_KEYS if k in src]
    if not have:
        raise SystemExit(f"{donor_ckpt} has no lm_head tensor — was it trained "
                         "with --tune lora? A LoRA run does not store the mT5.")
    w = src[have[0]]
    tgt = model.mt5.lm_head.weight
    if tuple(w.shape) != tuple(tgt.shape):
        raise SystemExit(f"lm_head shape {tuple(w.shape)} != {tuple(tgt.shape)}")
    before = float(torch.nn.functional.cosine_similarity(
        tgt.detach().double().flatten(), w.double().flatten(), dim=0))
    with torch.no_grad():
        tgt.copy_(w.to(tgt.dtype).to(tgt.device))
    return {"donor": str(donor_ckpt), "donor_run": donor_ckpt.parent.name,
            "cos_recipient_donor": round(before, 5),
            "params_M": round(w.numel() / 1e6, 1)}


class UniSignEval:
    """Uni-Sign with the discourse-context path (`tsl.unisign`)."""

    def __init__(self, ckpt: Path, ck: dict, posthoc_head: Path | None = None):
        self.model, _ = us.load_run(ckpt)
        self.graft = None
        if posthoc_head is not None:
            self.graft = graft_head(self.model, posthoc_head)
            print(f"  post-hoc lm_head <- {self.graft['donor_run']} "
                  f"({self.graft['params_M']}M, cos to recipient's own "
                  f"{self.graft['cos_recipient_donor']:+.3f})")
        self.model = self.model.cuda().eval()
        cfg = ck["cfg"]
        self.max_frames = cfg.get("max_frames", 192)
        self.ctx_max_frames = cfg.get("ctx_max_frames", 192)
        self.tags = {"init": cfg.get("init"), "tune": cfg.get("tune"),
                     "posthoc_head": self.graft}

    def dataset_kwargs(self) -> dict:
        return {"max_frames": self.max_frames, "ctx_max_frames": self.ctx_max_frames}

    def collate(self, batch):
        return us.collate_unisign(batch, self.max_frames, self.ctx_max_frames)

    def score(self, batch, cands):
        b = to_cuda(batch)
        return self.model.score(b["src"], cands, context_src=b["context_src"],
                                context_text=b["context_text"])

    def generate(self, batch):
        b = to_cuda(batch)
        return self.model.generate(b["src"], context_src=b["context_src"],
                                   context_text=b["context_text"])


class ScratchEval:
    """The from-scratch encoder + LoRA-LLM (`tsl.model`)."""

    def __init__(self, ckpt: Path, ck: dict):
        from tsl.model import ContextTranslator, ModelConfig

        cfg = ModelConfig(**ck["cfg"])
        model = ContextTranslator(cfg)
        model.load_adapter_state(ck["state"])
        self.model = model.cuda().eval()
        self.d_in = cfg.d_in
        self.max_frames = cfg.max_frames
        self.tags = {"fusion": cfg.fusion}

    def dataset_kwargs(self) -> dict:
        return {"max_frames": self.max_frames}

    def collate(self, batch):
        return collate(batch, self.d_in)

    def score(self, batch, cands):
        return self.model.score(to_cuda(batch), cands)

    def generate(self, batch):
        return self.model.generate(to_cuda(batch))


# ------------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS),
                    help="comma-separated. A `swap_plain` spec may carry a draw "
                         "index — `swap_plain@0,swap_plain@1,swap_plain@2` scores "
                         "three independent wrong-clip assignments in one pass, "
                         "and the rows record the spec, not the bare condition.")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", choices=["off", "bf16"], default="off")
    ap.add_argument("--generate", action="store_true",
                    help="also free-decode (slow); needed for BLEU/chrF and slot acc")
    ap.add_argument("--gen-conditions", default=None,
                    help="decode only these (default: every scored condition). "
                         "Beam search over the full test split costs far more "
                         "than the contrastive pass, and BLEU is only read off "
                         "one condition per system.")
    ap.add_argument("--items-only", action="store_true",
                    help="score only diagnostic items, skipping plain translation")
    ap.add_argument("--ctx-k", type=int, default=None,
                    help="override the training context window (for the "
                         "context-distance ablation)")
    ap.add_argument("--posthoc-head", type=Path, default=None,
                    help="another run's best.pt: graft its FINE-TUNED lm_head "
                         "onto this one and score with no retraining. The "
                         "§3.4 swaps happen at initialization and so cannot say "
                         "whether a mismatched readout blocks learning or only "
                         "hides it; this one can.")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    # mmap, and note this is the SECOND read of the same 7.8 GB file: `UniSignEval`
    # calls `us.load_run`, which loads it again to build the model. Both are
    # memory-mapped, so the two together cost pages rather than 16 GB of RAM.
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
    arch = ck.get("arch", "scratch")
    targs = ck["args"]
    fold = targs["fold"]
    ctx_k = a.ctx_k if a.ctx_k is not None else targs.get("ctx_k", 4)
    feat = Path(targs["feat"])

    if arch != "unisign" and a.posthoc_head is not None:
        raise SystemExit("--posthoc-head is a Uni-Sign intervention; this run is "
                         f"arch={arch!r}")
    system = (UniSignEval(a.ckpt, ck, posthoc_head=a.posthoc_head)
              if arch == "unisign" else ScratchEval(a.ckpt, ck))
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp == "bf16")

    records = load_records(a.data / "records.jsonl")
    folds = load_folds(a.data / "folds.json")
    test = split_records(records, folds, fold)["test"]
    donors = assign_donors(records, seed=targs.get("seed", 0))

    items = [r for r in test if r["item_type"] and r["candidates"]]
    print(f"{a.ckpt.parent.name}: arch {arch}, fold {fold}, {len(test)} test "
          f"utterances, {len(items)} contrastive items "
          f"({sum(r['item_type']=='anaphoric' for r in items)} anaphoric)")

    # Wrong-clip donors are drawn from this fold's own test split, so a swapped-in
    # clip is held out exactly as the target is, and never from the same
    # paragraph (`assign_target_swaps`). One map per draw index, built once.
    swap_maps: dict[tuple[str, int], dict[str, str]] = {}

    def swaps_for(cond: str, perm: int) -> dict[str, str]:
        key = (cond, perm)
        if key not in swap_maps:
            # seed=0 always, NOT the training seed: the wrong-clip assignment
            # is a property of the evaluation, not of the model, and keying it on
            # the training seed would hand a seed-1 rerun a different set of
            # wrong clips from the seed-0 cell it is being compared with.
            fn = assign_within_swaps if cond in SWAP_WITHIN else assign_target_swaps
            swap_maps[key] = fn(test, seed=0, perm=perm, pool=test)
        return swap_maps[key]

    def parse_cond(spec: str) -> tuple[str, int]:
        name, _, draw = spec.partition("@")
        if name not in CONDITIONS:
            raise SystemExit(f"unknown condition {name!r}; choose from {CONDITIONS}")
        if draw and name not in SWAP_TARGET:
            raise SystemExit(f"a draw index is only meaningful for "
                             f"{sorted(SWAP_TARGET)}, not {name!r}")
        return name, int(draw or 0)

    def loader(recs, cond, perm=0):
        ds = TSLDataset(recs, feat, condition=cond, ctx_k=ctx_k, donors=donors,
                        all_records=records,
                        swaps=swaps_for(cond, perm) if cond in SWAP_TARGET else None,
                        **system.dataset_kwargs())
        return DataLoader(ds, batch_size=a.batch, shuffle=False,
                          num_workers=a.workers, collate_fn=system.collate)

    rows: list[dict] = []
    gen_rows: list[dict] = []
    for spec in a.conditions.split(","):
        cond, perm = parse_cond(spec)
        n_hit = 0
        n = 0
        for batch in loader(items, cond, perm):
            meta = batch["meta"]
            cands = [m["candidates"] for m in meta]
            with torch.no_grad(), amp:
                scores = system.score(batch, cands)
            hits_mean = contrastive_accuracy(scores, "mean")
            hits_sum = contrastive_accuracy(scores, "sum")
            marg = margin(scores, "mean")
            picks = choice(scores, "mean")
            for m, hm, hs, mg, pk, sc in zip(meta, hits_mean, hits_sum, marg,
                                             picks, scores):
                rows.append({
                    "key": m["key"], "condition": spec, "item_type": m["item_type"],
                    # Which candidate, not merely whether it was gold: two models
                    # can agree on hit/miss item by item and still pick different
                    # distractors, which is the difference between "the same error
                    # pattern" and "the same solution".
                    "pick": pk, "tgt_src": m["tgt_src"], "tgt_ok": m["tgt_ok"],
                    "scores": [round(float(c["mean"]), 6) for c in sc],
                    "pron_gloss": m["pron_gloss"], "n_ctx": m["n_ctx"],
                    "agreeing_verb": bool(m["agreeing_verb"]),
                    "prior_adv": m["prior_correct"] is False,
                    "ctx_dist": m["ctx_dist"], "n_candidates": m["n_candidates"],
                    "hit": hm, "hit_sum": hs, "margin": mg,
                    "n_cand": len(sc),
                })
            n_hit += sum(hits_mean)
            n += len(hits_mean)
        note = ""
        if cond in SWAP_TARGET:
            m = swaps_for(cond, perm)
            got = sum(1 for it in items if it["key"] in m)
            if got != len(items):
                note = f"  [{got}/{len(items)} items had a donor]"
        print(f"  {spec:18s} contrastive {n_hit}/{n} = {n_hit/max(1,n):.1%}{note}",
              flush=True)

        if a.generate and (a.gen_conditions is None
                           or spec in a.gen_conditions.split(",")):
            gen_source = items if a.items_only else test
            for batch in loader(gen_source, cond, perm):
                meta = batch["meta"]
                refs = batch["answer"]
                with torch.no_grad(), amp:
                    hyps = system.generate(batch)
                for m, h, r in zip(meta, hyps, refs):
                    g = m["pron_gloss"]
                    slot = (pronoun_slot_hit(h, g, ALTERNATES.get(g, []))
                            if g and m["item_type"] else None)
                    gen_rows.append({
                        "key": m["key"], "condition": spec, "hyp": h, "ref": r,
                        "item_type": m["item_type"], "pron_gloss": g,
                        "slot_hit": slot,
                    })
            print(f"  {spec:18s} generated {len(gen_source)}", flush=True)

    out = a.out or a.ckpt.parent / "eval.json"
    out.write_text(json.dumps({
        "run": a.ckpt.parent.name, "arch": arch, "fold": fold, "cfg": ck["cfg"],
        "system": targs["system"], "backend": targs["backend"],
        "fusion": targs.get("fusion"), "seed": targs.get("seed", 0), "ctx_k": ctx_k,
        **system.tags,
        "contrastive": rows, "generation": gen_rows,
    }, ensure_ascii=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
