#!/usr/bin/env python3
"""What does the output projection do to the gradient the visual branch sees?

ReviewerO's P4, and the one measurement that would turn the paper's central claim
from an inference into a mechanism. The paper shows that a mismatched `lm_head`
at initialisation leaves the model at chance and that installing the same tensor
after fine-tuning rescues nothing, and concludes that the projection stops a
usable visual representation from FORMING. That is a claim about the training
trajectory, and so far every piece of evidence for it is an endpoint: an accuracy
after twelve epochs, or the absence of a rescue. Nothing has been measured about
the trajectory itself.

At step 0 it can be, and cheaply. Build the model exactly as training would --
the released visual half, the released mT5 half, one of four output projections
cross-loaded -- and take one backward pass on real training batches before any
weight has moved. Then read off how much gradient signal reaches the visual side,
and how coherent it is:

    |g| per group      pose branch, the pose->mT5 projection (`pose_proj`, the
                       6.9M half's actual interface), the mT5 body, `lm_head`
    share              the visual side's fraction of the total gradient norm --
                       "how much of what the loss wants is about the video"
    d loss / d pose-embeddings
                       the gradient at the boundary itself, which is independent
                       of how the two sides happen to be parameterised
    across-batch cos   mean pairwise cosine of the pose-branch gradient over
                       independent batches. A large but INCOHERENT gradient and a
                       small one are different pathologies: the first is noise the
                       optimiser averages away, the second is a signal that is not
                       there. Endpoint accuracies cannot tell them apart and this
                       can.

Three outcomes, three different sentences in the paper. If the ASL head shrinks
the visual-side gradient, "a mismatched projection starves the visual branch" is
a mechanism and not a metaphor. If it leaves the magnitude alone but destroys the
coherence, the mechanism is misdirection rather than starvation. If neither moves,
the trajectory account loses its most natural mechanism and the paper should say
that the effect is real and its cause unlocated -- which is worth knowing before
a reviewer asks a fourth time.

    ../.venv/bin/python scripts/30_grad_probe.py --init how2sign
    ../.venv/bin/python scripts/30_grad_probe.py --init csl_daily

-> data/grad_probe.json. No training and no optimizer state, but a backward pass
through the full 582M mT5, so `--grad-ckpt` (default on) keeps it inside the
memory a scoring pass uses rather than the memory a training cell uses.
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
from tsl.conditions import assign_donors, load_records, split_records  # noqa: E402
from tsl.dataset import TSLDataset  # noqa: E402
from tsl.splits import load as load_folds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# The levels of the output-projection factor, named as the training cells name
# them. `own` is the recipient's released head, i.e. the cell the paper reports
# as collapsed; the rest are the interventions.
LEVELS = ["own", "mt5_base", "csl_daily", "rand_head_nm", "how2sign"]


def to_cuda(b):
    if torch.is_tensor(b):
        return b.cuda()
    if isinstance(b, dict):
        return {k: to_cuda(v) for k, v in b.items() if k != "meta"}
    return b


def groups(model) -> dict[str, list]:
    """The same partition 04_train.py's optimizer uses, plus `lm_head` split out.

    `pose_proj` is separated from the rest of the pose branch because it is the
    one place the two halves touch: a projection that starves the visual side
    would show it here first.
    """
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    g: dict[str, list] = {"pose": [], "pose_proj": [], "lm_head": [], "mt5_body": []}
    for n, p in named:
        if n.startswith("core.mt5_model.lm_head"):
            g["lm_head"].append((n, p))
        elif n.startswith("core.mt5_model"):
            g["mt5_body"].append((n, p))
        elif "pose_proj" in n:
            g["pose_proj"].append((n, p))
        else:
            g["pose"].append((n, p))
    return g


def gnorm(pairs) -> float:
    s = 0.0
    for _, p in pairs:
        if p.grad is not None:
            s += float(p.grad.detach().double().pow(2).sum())
    return s ** 0.5


def gflat(pairs) -> torch.Tensor:
    return torch.cat([(p.grad.detach().float().flatten() if p.grad is not None
                       else torch.zeros(p.numel()))
                      for _, p in pairs])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="how2sign",
                    help="the checkpoint whose visual half and mT5 body are used")
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--batches", type=int, default=12,
                    help="independent batches; the across-batch cosine needs "
                         "several and its variance falls slowly")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--label-smoothing", type=float, default=0.2,
                    help="upstream's value, so the loss is the training loss")
    ap.add_argument("--max-frames", type=int, default=192)
    ap.add_argument("--grad-ckpt", action="store_true", default=True)
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "grad_probe.json")
    a = ap.parse_args()

    records = load_records(a.data / "records.jsonl")
    folds = load_folds(a.data / "folds.json")
    train = split_records(records, folds, a.fold)["train"]
    donors = assign_donors(records, seed=a.seed)
    feat = a.data / "feat_rtmpose"
    print(f"fold {a.fold}: {len(train)} train utterances, "
          f"{a.batches} batches of {a.batch}")

    # One fixed batch order for every level, so the comparison is between
    # projections and not between draws.
    ds = TSLDataset(train, feat, condition="local", ctx_k=4, donors=donors,
                    all_records=records, max_frames=a.max_frames,
                    ctx_max_frames=a.max_frames)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    generator=torch.Generator().manual_seed(a.seed),
                    collate_fn=lambda b: us.collate_unisign(b, a.max_frames,
                                                            a.max_frames))
    batches = []
    for i, b in enumerate(dl):
        if i >= a.batches:
            break
        batches.append(b)

    out = {}
    if a.out.exists():
        out = json.loads(a.out.read_text())

    for level in [x.strip() for x in a.levels.split(",") if x.strip()]:
        if level == a.init:
            continue  # "own" already covers it
        print(f"\n=== {a.init} + {level} lm_head")
        ck = us.init_path(a.init, a.data / "external")
        ck2 = None if level == "own" else us.init_path(level, a.data / "external")
        model = us.UniSignTSL(checkpoint=ck, lang="Chinese",
                              label_smoothing=a.label_smoothing,
                              parts="all", checkpoint_mt5=ck2,
                              parts_mt5="lm_head")
        if a.grad_ckpt:
            model.mt5.gradient_checkpointing_enable()
        model = model.cuda().train()
        g = groups(model)
        print("   " + "  ".join(f"{k} {sum(p.numel() for _, p in v)/1e6:.1f}M"
                                for k, v in g.items()))

        rows, pose_g, losses = [], [], []
        for bi, batch in enumerate(batches):
            model.zero_grad(set_to_none=True)
            b = to_cuda(batch)
            # Gradient at the interface itself, read off the pose embeddings
            # rather than off any parameter: it does not depend on how either
            # side is parameterised, so it is the one comparable quantity if the
            # two halves ever differ in shape.
            embeds, mask = model.encoder_inputs(b["src"], None, None)
            embeds.retain_grad()
            lab = model.tok(batch["answer"], return_tensors="pt", padding=True,
                            truncation=True,
                            max_length=model.max_target_len)["input_ids"]
            lab[lab == model.tok.pad_token_id] = -100
            lab = lab.to(embeds.device)
            o = model.mt5(inputs_embeds=embeds, attention_mask=mask, labels=lab,
                          return_dict=True)
            loss = torch.nn.functional.cross_entropy(
                o.logits.reshape(-1, o.logits.shape[-1]).float(),
                lab.reshape(-1), ignore_index=-100,
                label_smoothing=a.label_smoothing)
            loss.backward()
            n_pose = b["src"]["attention_mask"].shape[1]
            emb_g = embeds.grad
            row = {"batch": bi, "loss": float(loss),
                   "g_pose": gnorm(g["pose"]),
                   "g_pose_proj": gnorm(g["pose_proj"]),
                   "g_lm_head": gnorm(g["lm_head"]),
                   "g_mt5_body": gnorm(g["mt5_body"]),
                   "g_embeds_all": float(emb_g.detach().double().norm()),
                   "g_embeds_pose": float(
                       emb_g[:, -n_pose:].detach().double().norm())}
            row["g_total"] = (row["g_pose"] ** 2 + row["g_pose_proj"] ** 2
                              + row["g_lm_head"] ** 2 + row["g_mt5_body"] ** 2) ** 0.5
            row["visual_share"] = ((row["g_pose"] ** 2 + row["g_pose_proj"] ** 2)
                                   ** 0.5 / max(row["g_total"], 1e-12))
            rows.append(row)
            losses.append(float(loss))
            pose_g.append(gflat(g["pose"] + g["pose_proj"]).cpu())
            print(f"   b{bi:02d} loss {row['loss']:.3f}  |g|pose "
                  f"{row['g_pose']:.4g}  proj {row['g_pose_proj']:.4g}  "
                  f"head {row['g_lm_head']:.4g}  body {row['g_mt5_body']:.4g}  "
                  f"emb(pose) {row['g_embeds_pose']:.4g}", flush=True)

        # Coherence: do independent batches agree about which way the visual
        # branch should move? A magnitude alone cannot answer that.
        cos = []
        for i in range(len(pose_g)):
            for j in range(i + 1, len(pose_g)):
                cos.append(float(torch.nn.functional.cosine_similarity(
                    pose_g[i][None].double(), pose_g[j][None].double(), dim=1)))
        mean = lambda k: sum(r[k] for r in rows) / len(rows)  # noqa: E731
        rec = {"init": a.init, "head": level, "fold": a.fold,
               "batch": a.batch, "n_batches": len(rows),
               "params_M": {k: sum(p.numel() for _, p in v) / 1e6
                            for k, v in g.items()},
               "loss": mean("loss"),
               "g_pose": mean("g_pose"), "g_pose_proj": mean("g_pose_proj"),
               "g_lm_head": mean("g_lm_head"), "g_mt5_body": mean("g_mt5_body"),
               "g_total": mean("g_total"), "visual_share": mean("visual_share"),
               "g_embeds_pose": mean("g_embeds_pose"),
               "g_embeds_all": mean("g_embeds_all"),
               "pose_grad_cos_mean": sum(cos) / max(len(cos), 1),
               "pose_grad_cos_min": min(cos) if cos else None,
               "pose_grad_cos_max": max(cos) if cos else None,
               "rows": rows}
        print(f"   -> loss {rec['loss']:.4f}  visual share "
              f"{rec['visual_share']:.4f}  pose-grad coherence "
              f"{rec['pose_grad_cos_mean']:.4f}")
        out[f"{a.init}+{level}"] = rec
        a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        del model
        torch.cuda.empty_cache()

    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
