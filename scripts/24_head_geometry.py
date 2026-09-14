#!/usr/bin/env python3
"""What is actually different about the checkpoints' output projections.

Appendix D reports cosines *between* checkpoints' `lm_head` tensors: 0.95 inside
the CSL pair, 0.86-0.95 inside the ASL group, 0.27-0.31 across. Two reviews point
out that this is not enough to support the reading the paper puts on it, for two
different reasons, and both are cheap to answer.

1. Distance to the common ancestor is missing (`--to-base`).
   Every one of these heads started as `google/mt5-base`'s. A within-group cosine
   of 0.95 and a between-group cosine of 0.29 is consistent with "two output
   languages" and equally consistent with "one group barely moved and the other
   moved a long way in an arbitrary direction". Those are different papers. The
   distance from each head to mT5-base's separates them, and it is a ten-minute
   计算 that can change §3.4's direction.

   Reported two ways, because a review asks which one Appendix D meant: `cos` on
   the flattened 250112x768 tensor, and `row_cos` as the mean over the 250112
   per-token rows. They are not the same statistic and can disagree -- a flat
   cosine is dominated by the high-norm rows.

2. The head may only be rescaling Chinese rows (`--cjk`).
   If CSL fine-tuning systematically raises the norm of rows for tokens
   containing a Han character and lowers the rest, then "the output projection
   carries the written language" is close to a tautology, and the paper's real
   claim has to move to how that conditions visual transfer. Partition the rows
   by whether the token contains a CJK unified ideograph and measure: mean row
   norm per group, the CJK/non-CJK ratio, and how each group moved from mT5-base.

    ../.venv/bin/python scripts/24_head_geometry.py     # -> data/head_geometry.json

CPU only. One checkpoint in memory at a time -- the box is shared and each file
is 1.2 GB on disk but 966.6M mT5 parameters when the mT5 half is materialized.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
FILES = {
    "csl_stage1": "csl_stage1_weight.pth",
    "csl_daily": "csl_daily_pose_only_slt.pth",
    "how2sign": "how2sign_pose_only_slt.pth",
    "openasl": "openasl_pose_only_slt.pth",
    "wlasl": "wlasl_pose_only_islr.pth",
    # Both written by us, and neither is a sixth pretrained model: mt5_base is the
    # published mT5-base (scripts/22_mt5base_ckpt.py) and tsl_text is mT5-base
    # after text-only Chinese adaptation with no sign language anywhere in it
    # (scripts/23_text_lm.py). They are the two neutral reference points the
    # released files cannot provide.
    "mt5_base": "mt5_base.pth",
    "tsl_text": "tsl_text_lm.pth",
}
HAN = re.compile(r"[一-鿿㐀-䶿]")
# The symmetric group, added 8/22. "An ASL fine-tune attenuates Chinese rows
# harder than the rest" invites the reply that it attenuates everything and CJK
# rows merely started higher. Splitting the non-CJK side into Latin-script rows
# and the remainder answers it: if the two groups are attenuated in OPPOSITE
# orders by the two checkpoint families, the effect is selective rather than a
# global rescaling with a different intercept.
LATIN = re.compile(r"[A-Za-z]")
# mT5's SentencePiece marks a word start with U+2581; strip it before asking
# whether the piece contains a Han character, or every word-initial Chinese piece
# would still match but the boundary marker would count as content.
SPM_SPACE = "▁"


def head(path: Path) -> torch.Tensor | None:
    obj = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = obj
    for key in ("model", "state_dict", "module"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    for k, v in sd.items():
        if k.replace("module.", "", 1) == "mt5_model.lm_head.weight":
            return v.detach().float()
    return None


def flat_cos(a: torch.Tensor, b: torch.Tensor) -> dict:
    x, y = a.double().flatten(), b.double().flatten()
    return {
        "cos": float(torch.dot(x, y) / (x.norm() * y.norm())),
        "rel_l2": float((x - y).norm() / y.norm()),
    }


def row_cos(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Per-token cosine, then averaged. The flat cosine over 192.1M parameters is
    a norm-weighted average of these, so a handful of very frequent tokens can
    carry it; the unweighted mean says whether the whole vocabulary moved."""
    c = torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=1)
    q = torch.quantile(c, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
    return {"row_cos_mean": float(c.mean()), "row_cos_p25": float(q[0]),
            "row_cos_median": float(q[1]), "row_cos_p75": float(q[2])}


def cjk_mask(vocab_rows: int) -> tuple[torch.Tensor, dict, torch.Tensor]:
    from transformers import T5Tokenizer

    tok = T5Tokenizer.from_pretrained("google/mt5-base", legacy=False)
    toks = tok.convert_ids_to_tokens(list(range(min(vocab_rows, len(tok)))))
    m = torch.zeros(vocab_rows, dtype=torch.bool)
    lat = torch.zeros(vocab_rows, dtype=torch.bool)
    for i, t in enumerate(toks):
        if not t:
            continue
        bare = t.replace(SPM_SPACE, "")
        if HAN.search(bare):
            m[i] = True
        elif LATIN.search(bare):
            lat[i] = True
    # Rows past the tokenizer's length exist because mT5 pads its vocabulary to a
    # multiple of 128. They are neither CJK nor non-CJK content and are counted
    # separately rather than lumped into the non-CJK group, where 12 dead rows
    # would be invisible but a larger pad would not.
    info = {"vocab_rows": vocab_rows, "tokenizer_len": len(tok),
            "cjk_rows": int(m.sum()), "latin_rows": int(lat.sum()),
            "pad_rows": max(0, vocab_rows - len(tok))}
    return m, info, lat


def norms(w: torch.Tensor, m: torch.Tensor, n_tok: int,
          lat: torch.Tensor | None = None) -> dict:
    rn = w.double().norm(dim=1)
    live = torch.zeros_like(m)
    live[:n_tok] = True
    cjk, other = m & live, (~m) & live
    out = {"cjk_mean_norm": float(rn[cjk].mean()),
           "other_mean_norm": float(rn[other].mean()),
           "ratio_cjk_other": float(rn[cjk].mean() / rn[other].mean()),
           "pad_mean_norm": (float(rn[~live].mean()) if int((~live).sum()) else None)}
    if lat is not None:
        la = lat & live
        rest = live & (~m) & (~lat)
        out["latin_mean_norm"] = float(rn[la].mean())
        out["rest_mean_norm"] = float(rn[rest].mean())
        out["ratio_cjk_latin"] = float(rn[cjk].mean() / rn[la].mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "head_geometry.json")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="checkpoint names to leave out (a file not written yet)")
    a = ap.parse_args()

    names = [n for n in FILES if n not in a.skip and (EXT / FILES[n]).exists()]
    absent = [n for n in FILES if n not in names]
    if absent:
        print(f"absent, skipped: {absent}")
    if "mt5_base" not in names:
        raise SystemExit("mt5_base.pth is the reference point; run "
                         "scripts/22_mt5base_ckpt.py first")

    base = head(EXT / FILES["mt5_base"])
    if base is None:
        raise SystemExit("no lm_head in mt5_base.pth")
    mask, info, latin = cjk_mask(base.shape[0])
    n_tok = info["tokenizer_len"]
    print(f"{info['vocab_rows']} rows: {info['cjk_rows']} contain a Han character, "
          f"{info['latin_rows']} a Latin letter and no Han, "
          f"{info['pad_rows']} are vocabulary padding")

    out: dict = {"vocab": info, "to_mt5base": {}, "row_norms": {}}
    out["row_norms"]["mt5_base"] = norms(base, mask, n_tok, latin)
    for n in names:
        if n == "mt5_base":
            continue
        w = head(EXT / FILES[n])
        if w is None:
            print(f"{n}: no lm_head, skipped")
            continue
        d = {**flat_cos(w, base), **row_cos(w, base)}
        out["to_mt5base"][n] = d
        rn = norms(w, mask, n_tok, latin)
        b = out["row_norms"]["mt5_base"]
        rn["cjk_growth"] = rn["cjk_mean_norm"] / b["cjk_mean_norm"]
        rn["other_growth"] = rn["other_mean_norm"] / b["other_mean_norm"]
        rn["latin_growth"] = rn["latin_mean_norm"] / b["latin_mean_norm"]
        rn["rest_growth"] = rn["rest_mean_norm"] / b["rest_mean_norm"]
        out["row_norms"][n] = rn
        print(f"{n:11s} to mt5-base: cos {d['cos']:+.4f}  rel_l2 {d['rel_l2']:.4f}  "
              f"row_cos {d['row_cos_mean']:+.4f} (median {d['row_cos_median']:+.4f})")
        print(f"{'':11s}   growth: CJK x{rn['cjk_growth']:.3f}  Latin "
              f"x{rn['latin_growth']:.3f}  rest x{rn['rest_growth']:.3f}   "
              f"CJK/Latin norm ratio {rn['ratio_cjk_latin']:.3f}")
        del w

    b = out["row_norms"]["mt5_base"]
    print(f"{'mt5_base':11s}   CJK/Latin norm ratio {b['ratio_cjk_latin']:.3f} "
          f"(CJK {b['cjk_mean_norm']:.3f}, Latin {b['latin_mean_norm']:.3f}, "
          f"rest {b['rest_mean_norm']:.3f})")
    a.out.write_text(json.dumps(out, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
