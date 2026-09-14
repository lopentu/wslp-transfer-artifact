"""Records + precomputed features -> batches, one experimental condition at a time.

The conditions themselves, and the mismatched-context donor assignment, live in
`tsl.conditions` (torch-free, unit-tested). This module is only the tensor side.
"""

from __future__ import annotations

import random
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .conditions import (  # re-exported for convenience
    BLANK_TARGET, CONDITIONS, CTX_TEXT, CTX_VIDEO, MISMATCH_TEXT, MISMATCH_VIDEO,
    PROMPT_CTX, PROMPT_PLAIN, SHUFFLE_FRAMES, SWAP_TARGET, assign_donors,
    assign_target_swaps, context_text, last_k, load_records, prompt_for,
    split_records,
)

__all__ = [
    "TSLDataset", "collate", "CONDITIONS", "assign_donors", "assign_target_swaps",
    "load_records", "split_records",
]


@lru_cache(maxsize=96)
def _load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


class TSLDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        feat_dir: Path,
        condition: str | dict[str, float] = "both",
        max_frames: int = 256,
        ctx_max_frames: int = 256,
        ctx_k: int | None = 4,
        seed: int = 0,
        donors: dict[str, str] | None = None,
        all_records: list[dict] | None = None,
        swaps: dict[str, str] | None = None,
    ):
        self.recs = records
        self.feat_dir = Path(feat_dir)
        self.condition = condition
        self.max_frames = max_frames
        self.ctx_max_frames = ctx_max_frames
        # How many preceding utterances to actually use. Records store 8; video is
        # additionally capped by ctx_max_frames, so a long text window can pair
        # with a shorter video one rather than blowing up the visual sequence.
        self.ctx_k = ctx_k
        self.seed = seed
        pool = all_records if all_records is not None else records
        self.by_key = {r["key"]: r for r in pool}
        self.siblings: dict[str, list[str]] = {}
        for r in pool:
            self.siblings.setdefault(r["uuid"], []).append(r["key"])
        self.donors = donors if donors is not None else assign_donors(pool, seed)
        # Target-clip donors, for `swap_plain`. Left None unless the caller
        # supplies them: the pool a swapped-in clip should come from is the
        # evaluated split, which this object cannot know — `05_eval.py` passes the
        # test records. A `swap_plain` batch without them would silently be a
        # `local` batch, so `__getitem__` refuses instead.
        self.swaps = swaps

    def __len__(self) -> int:
        return len(self.recs)

    def _cond(self, i: int) -> str:
        if isinstance(self.condition, str):
            return self.condition
        # Deterministic per index, so a dev pass is reproducible across epochs and
        # the mix is a fixed property of the dataset rather than of the shuffle.
        r = np.random.default_rng(self.seed * 1_000_003 + i)
        names = list(self.condition)
        w = np.array([self.condition[n] for n in names], dtype=float)
        return names[int(r.choice(len(names), p=w / w.sum()))]

    # ------------------------------------------------------------------ features

    def _feat(self, key: str) -> np.ndarray | None:
        uuid = key.split(":")[0]
        p = self.feat_dir / f"{uuid}.npz"
        if not p.exists():
            return None
        return _load_npz(str(p)).get(key)

    @staticmethod
    def _subsample(x: np.ndarray, cap: int) -> np.ndarray:
        if len(x) <= cap:
            return x
        idx = np.linspace(0, len(x) - 1, cap).round().astype(int)
        return x[idx]

    def _context_frames(self, rec: dict, cond: str) -> np.ndarray | None:
        if cond in MISMATCH_VIDEO:
            donor = self.by_key.get(self.donors.get(rec["key"], ""))
            if donor is None:
                return None
            dctx = last_k(donor["ctx"], self.ctx_k)
            src = [f"{donor['uuid']}:{c['seq']:03d}" for c in dctx] or [donor["key"]]
        elif cond == "shuffle_video":
            exclude = {rec["key"]} | {f"{rec['uuid']}:{c['seq']:03d}"
                                      for c in last_k(rec["ctx"], self.ctx_k)}
            sibs = sorted(set(self.siblings.get(rec["uuid"], [])) - exclude)
            if not sibs:
                return None
            # crc32, not hash(): builtin str hash is salted per process, so the
            # shuffled condition would mean something different on every run.
            src = [random.Random(zlib.crc32(rec["key"].encode())).choice(sibs)]
        elif cond in CTX_VIDEO:
            src = [f"{rec['uuid']}:{c['seq']:03d}" for c in last_k(rec["ctx"], self.ctx_k)]
        else:
            return None
        parts = [f for f in (self._feat(k) for k in src) if f is not None and len(f)]
        if not parts:
            return None
        return self._subsample(np.concatenate(parts, 0), self.ctx_max_frames)

    def __getitem__(self, i: int) -> dict:
        rec = self.recs[i]
        cond = self._cond(i)
        tgt = self._feat(rec["key"])
        if tgt is None or len(tgt) == 0:
            tgt = np.zeros((1, 1), np.float16)  # collate flags this invalid
            ok = False
        else:
            tgt = self._subsample(tgt, self.max_frames)
            ok = True
        tgt_src = rec["key"]
        if cond in BLANK_TARGET:
            tgt = np.zeros_like(tgt)
        elif cond in SWAP_TARGET:
            if self.swaps is None:
                raise ValueError(
                    f"condition {cond!r} needs a target-clip donor map; pass "
                    "swaps=assign_target_swaps(...) to TSLDataset")
            donor_key = self.swaps.get(rec["key"])
            donor = self._feat(donor_key) if donor_key else None
            if donor is None or len(donor) == 0:
                # Never fall through to the true clip: that would report a
                # swapped cell that was really `local`. Flagged instead, and
                # 05_eval.py drops these items from the swapped condition.
                ok = False
            else:
                tgt = self._subsample(donor, self.max_frames)
                tgt_src = donor_key
        elif cond in SHUFFLE_FRAMES:
            # crc32 for the same reason `shuffle_video` uses it: builtin str hash
            # is salted per process, so the condition would differ between runs.
            idx = np.arange(len(tgt))
            random.Random(zlib.crc32(rec["key"].encode())).shuffle(idx)
            tgt = tgt[idx]

        ctx_txt = context_text(rec, cond, self.donors, self.by_key, self.ctx_k)
        return {
            "key": rec["key"],
            "condition": cond,
            "tgt_src": tgt_src,
            "tgt": tgt.astype(np.float32),
            "tgt_ok": ok,
            "ctx": self._context_frames(rec, cond),
            # Both forms of the context text: `prompt` is the assembled Chinese
            # instruction the from-scratch model.py expects, `ctx_text` is the
            # bare context Uni-Sign folds into its own English prefix. They must
            # come from the same call, or the two systems would silently be
            # reading different context.
            "prompt": prompt_for(ctx_txt),
            "ctx_text": ctx_txt,
            "answer": rec["text"],
            "candidates": rec.get("candidates") or [],
            "item_type": rec.get("item_type"),
            "pron_gloss": rec.get("pron_gloss"),
            "resolved": bool(rec.get("resolved_referent")),
            "prior_correct": rec.get("prior_correct"),
            "agreeing_verb": rec.get("agreeing_verb"),
            "ctx_dist": rec.get("ctx_dist"),
            "n_candidates": rec.get("n_candidates", 0),
            "n_ctx": len(last_k(rec["ctx"], self.ctx_k)),
        }


def collate(items: list[dict], d_in: int) -> dict:
    B = len(items)
    tt = max(max(len(i["tgt"]) for i in items), 1)
    ct = max((len(i["ctx"]) for i in items if i["ctx"] is not None), default=1)

    tgt = torch.zeros(B, tt, d_in)
    tgt_valid = torch.zeros(B, tt, dtype=torch.bool)
    ctx = torch.zeros(B, ct, d_in)
    ctx_valid = torch.zeros(B, ct, dtype=torch.bool)
    has_ctx = torch.zeros(B, dtype=torch.bool)

    for b, it in enumerate(items):
        x = it["tgt"]
        if it["tgt_ok"] and x.shape[-1] == d_in:
            n = len(x)
            tgt[b, :n] = torch.from_numpy(x)
            tgt_valid[b, :n] = True
        else:
            tgt_valid[b, 0] = True  # one live slot, so attention cannot see all-pad
        c = it["ctx"]
        if c is not None and c.shape[-1] == d_in and len(c):
            n = len(c)
            ctx[b, :n] = torch.from_numpy(c.astype(np.float32))
            ctx_valid[b, :n] = True
            has_ctx[b] = True
        else:
            ctx_valid[b, 0] = True

    return {
        "tgt_feat": tgt, "tgt_valid": tgt_valid,
        "ctx_feat": ctx, "ctx_valid": ctx_valid, "has_ctx": has_ctx,
        "prompt": [i["prompt"] for i in items],
        "answer": [i["answer"] for i in items],
        "meta": [{k: i[k] for k in
                  ("key", "condition", "tgt_src", "candidates", "item_type",
                   "pron_gloss", "resolved", "prior_correct", "agreeing_verb",
                   "ctx_dist", "n_candidates", "n_ctx", "tgt_ok")} for i in items],
    }
