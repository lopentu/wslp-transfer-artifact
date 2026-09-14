#!/usr/bin/env python3
"""Does the mismatched model's representation collapse, measured inside the model?

The paper's claim that a written-language-mismatched initialisation collapses onto
a video-independent prior is behavioural: the model scores at chance, gains
nothing from the clip, and answers most inputs with one string. A reviewer asked
for the internal version --- show that the representations themselves stop
distinguishing clips --- and this is that check, in a form a reader can verify
rather than eyeball.

Not UMAP. A two-dimensional projection of a 768-dimensional space is a picture
whose shape depends on the projection's hyperparameters, and "the points look
clumped" is not a quantity. Two scalars over the same representations say the
same thing and can be compared across models:

    participation ratio   PR = (sum lambda)^2 / sum lambda^2 over the covariance
                          eigenvalues of the per-utterance representations. This
                          is the effective number of dimensions the representation
                          actually uses: PR near 1 means every clip maps to
                          essentially the same vector, which is what collapse
                          means.
    mean pairwise cosine  of the centred representations. Near 1 for a collapsed
                          representation, near 0 for one that separates clips.

Measured at two points, because where the information is lost is itself a result:

    pose        the visual half's output (`pose_embeds`), mean-pooled over frames
    encoder     the mT5 encoder's last hidden state over the pose positions

The mT5 half contains the encoder that reads the pose embeddings (see the paper's
§2), so a model whose pose output still varies while its encoder output does not
has lost the clip on the text side, not in the visual front end.

    python3 scripts/20_repr_collapse.py --cells f0_local_csl_daily_full_k4_s0,...

One forward pass per utterance, no decoding and no retraining.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsl import unisign as us  # noqa: E402
from tsl.conditions import load_records, split_records  # noqa: E402
from tsl.dataset import TSLDataset  # noqa: E402
from tsl.splits import load as load_folds  # noqa: E402


def stats(x: np.ndarray) -> dict:
    """Participation ratio and mean pairwise cosine of a set of vectors."""
    x = x.astype(np.float64)
    xc = x - x.mean(0, keepdims=True)
    # Eigenvalues of the covariance, via the Gram matrix when that is smaller.
    cov = xc.T @ xc / max(1, len(xc) - 1)
    lam = np.linalg.eigvalsh(cov)
    lam = np.clip(lam, 0, None)
    pr = float(lam.sum() ** 2 / max(1e-30, (lam ** 2).sum()))
    n = xc / np.maximum(1e-12, np.linalg.norm(xc, axis=1, keepdims=True))
    g = n @ n.T
    iu = np.triu_indices(len(g), 1)
    return {"n": int(len(x)), "pr": pr, "cos": float(g[iu].mean()),
            "var": float(np.trace(cov))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="comma-separated run names")
    ap.add_argument("--limit", type=int, default=400,
                    help="test utterances per cell; the statistics are stable "
                         "well before this and the card is shared")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "repr_collapse.json")
    a = ap.parse_args()

    out: dict[str, dict] = {}
    if a.out.exists():
        out = json.loads(a.out.read_text())

    for name in a.cells.split(","):
        ck_path = ROOT / "runs" / name / "best.pt"
        if not ck_path.exists():
            print(f"skip {name} (not trained)")
            continue
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        targs = ck["args"]
        model, _ = us.load_run(ck_path)
        model = model.cuda().eval()

        records = load_records(ROOT / "data" / "records.jsonl")
        folds = load_folds(ROOT / "data" / "folds.json")
        test = split_records(records, folds, targs["fold"])["test"][: a.limit]
        ds = TSLDataset(test, Path(targs["feat"]), condition="local",
                        ctx_k=targs.get("ctx_k", 4), all_records=records,
                        max_frames=ck["cfg"].get("max_frames", 192),
                        ctx_max_frames=ck["cfg"].get("ctx_max_frames", 192))
        dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=4,
                        collate_fn=lambda b: us.collate_unisign(
                            b, ck["cfg"].get("max_frames", 192),
                            ck["cfg"].get("ctx_max_frames", 192)))

        pose_v, enc_v = [], []
        for batch in dl:
            src = {k: (v.cuda() if torch.is_tensor(v) else v)
                   for k, v in batch["src"].items()}
            with torch.no_grad():
                pose = model.pose_embeds(src)                 # B,T,768
                m = src["attention_mask"].unsqueeze(-1).float()
                pose_v.append(((pose * m).sum(1) / m.sum(1).clamp(min=1))
                              .float().cpu().numpy())
                embeds, mask = model.encoder_inputs(src, None, batch["context_text"])
                h = model.mt5.encoder(inputs_embeds=embeds, attention_mask=mask,
                                      return_dict=True).last_hidden_state
                # Pose positions only: the prompt prefix is identical across
                # utterances, so pooling it in would dilute exactly the variation
                # this script is measuring.
                npose = pose.shape[1]
                hp = h[:, -npose:]
                mm = src["attention_mask"].unsqueeze(-1).float()
                enc_v.append(((hp * mm).sum(1) / mm.sum(1).clamp(min=1))
                             .float().cpu().numpy())

        row = {"pose": stats(np.concatenate(pose_v)),
               "encoder": stats(np.concatenate(enc_v)),
               "init": ck["cfg"].get("init"), "fold": targs["fold"]}
        out[name] = row
        print(f"{name}")
        for k in ("pose", "encoder"):
            s = row[k]
            print(f"   {k:8s} PR {s['pr']:7.2f}  mean cos {s['cos']:+.3f}  "
                  f"trace {s['var']:.3f}  (n={s['n']})")
        a.out.write_text(json.dumps(out, indent=1))
        del model
        torch.cuda.empty_cache()

    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
