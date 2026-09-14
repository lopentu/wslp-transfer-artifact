#!/usr/bin/env python3
"""Train one system on one fold.

Two axes cross here, and the script keeps them independent so a run is one cell
of the design rather than a bundle of choices.

**--init — the transfer axis** (README §4, axis 1). Which Uni-Sign checkpoint the
weights start from: `csl_daily` / `csl_stage1` (CSL, Chinese output), `how2sign` /
`openasl` (ASL, English output), `wlasl` (ASL isolated signs), or `random` — same
architecture and the same pretrained mT5-base, with only the 6.9 M pose branch
left untrained. All five released checkpoints are size-matched at 973.5 M / 627
tensors, so a difference between rows is a difference in what the weights were
pretrained on and not in capacity.

**--system — the discourse axis** (axis 2):

    --system context   trained with a mix of conditions (context dropout), so a
                       SINGLE checkpoint can be evaluated under local / correct /
                       mismatched / shuffled at inference. That is the trick that
                       makes the full condition sweep affordable: the conditions
                       differ only in what is fed at test time, so any difference
                       between them cannot be a difference between models.

    --system local     trained with condition=local always — the no-context
                       ceiling. Needed because a context-trained model evaluated
                       under `local` is out of its training distribution, and the
                       gap would otherwise be read as a context effect. With no
                       context on either path this reproduces upstream Uni-Sign's
                       own forward exactly, so the baseline is faithful rather
                       than crippled.

`--tune lora` (default) trains the pose branch plus LoRA on the mT5 and leaves
the rest frozen — the plan of record, because full fine-tuning needs ~12.7 GB on
a shared card (README §5). `--tune full` is the opportunistic version; it also
saves a 2.4 GB checkpoint per run instead of ~60 MB.

`--arch scratch` keeps the earlier from-scratch encoder + LoRA-LLM design in
`tsl.model` reachable as the no-pretrained-sign-encoder ablation.

    python3 scripts/04_train.py --fold 0 --init csl_daily --system context
    python3 scripts/04_train.py --fold 0 --init csl_daily --system local
    python3 scripts/04_train.py --fold 0 --init random   --system context
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsl import unisign as us  # noqa: E402
from tsl.conditions import assign_donors, load_records, split_records  # noqa: E402
from tsl.dataset import TSLDataset, collate  # noqa: E402
from tsl.features import DINO_DIM, POSE_DIM, RTMPOSE_DIM  # noqa: E402
from tsl.splits import load as load_folds  # noqa: E402

# Context dropout schedule. `both` dominates so the fusion path is actually
# trained; `local` is well represented so the model does not collapse into
# requiring context; the single-modality views keep each path usable alone.
TRAIN_MIX = {"both": 0.45, "local": 0.25, "video": 0.15, "text": 0.15}


# --------------------------------------------------------------------- systems
#
# Both systems answer the same three questions — how to batch, what the loss is,
# and what to save — so the training loop below is written once.


class UniSignSystem:
    arch = "unisign"

    def __init__(self, a):
        if a.backend != "rtmpose":
            raise SystemExit("--arch unisign needs the rtmpose features "
                             "(Uni-Sign's own 133-keypoint format); "
                             f"got --backend {a.backend}")
        ckpt = us.init_path(a.init, a.data / "external")
        ckpt_mt5 = us.init_path(a.init_mt5, a.data / "external") if a.init_mt5 else None
        self.model = us.UniSignTSL(checkpoint=ckpt, lang="Chinese",
                                   label_smoothing=a.label_smoothing,
                                   parts=a.init_parts, checkpoint_mt5=ckpt_mt5,
                                   parts_mt5=a.init_mt5_parts)
        if self.model.loads:
            for r in self.model.loads:
                print(f"init {Path(r['checkpoint']).name} [{r['parts']}]: "
                      f"{r['loaded']} of {r['in_file']} tensors loaded, "
                      f"{r['missing']} missing, {r['unexpected']} unexpected")
                # A row that quietly failed to load its weights looks exactly
                # like "transfer does not help", so refuse to train one.
                if r["missing"] or r["unexpected"]:
                    raise SystemExit(f"checkpoint did not match the model: "
                                     f"missing {r['missing_sample']}, "
                                     f"unexpected {r['unexpected_sample']}")
        else:
            print("init random: pose branch untrained, mT5-base as published")
        if a.tune == "lora":
            self.model.apply_lora(r=a.lora_r, alpha=a.lora_alpha,
                                  dropout=a.lora_dropout)
        if a.grad_ckpt:
            self.model.mt5.gradient_checkpointing_enable()
        self.model.cuda()
        self.a = a

    def collate(self, batch):
        return us.collate_unisign(batch, self.a.max_frames, self.a.ctx_max_frames)

    def loss(self, batch, label_smoothing=None):
        b = to_cuda(batch)
        return self.model(b["src"], b["answer"], context_src=b["context_src"],
                          context_text=b["context_text"],
                          label_smoothing=label_smoothing)

    def groups(self, a) -> list[dict]:
        named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        lora = [p for n, p in named if "lora_" in n]
        mt5 = [p for n, p in named if "lora_" not in n and n.startswith("core.mt5_model")]
        pose = [p for n, p in named if "lora_" not in n
                and not n.startswith("core.mt5_model")]
        return [{"params": pose, "lr": a.lr_pose, "name": "pose"},
                {"params": lora, "lr": a.lr_lora, "name": "lora"},
                {"params": mt5, "lr": a.lr_mt5, "name": "mt5"}]

    def cfg(self, a) -> dict:
        ckpt = us.init_path(a.init, a.data / "external")
        ckpt_mt5 = us.init_path(a.init_mt5, a.data / "external") if a.init_mt5 else None
        return {"init": a.init, "init_parts": a.init_parts,
                "checkpoint": str(ckpt) if ckpt else "",
                "init_mt5": a.init_mt5, "init_mt5_parts": a.init_mt5_parts,
                "checkpoint_mt5": str(ckpt_mt5) if ckpt_mt5 else "",
                "lang": "Chinese", "label_smoothing": a.label_smoothing,
                "tune": a.tune, "lora_r": a.lora_r, "lora_alpha": a.lora_alpha,
                "lora_dropout": a.lora_dropout, "max_target_len": 64,
                "max_frames": a.max_frames, "ctx_max_frames": a.ctx_max_frames}

    def state(self) -> dict:
        return self.model.trainable_state()

    def extra(self) -> dict:
        return {}


class ScratchSystem:
    """The earlier from-scratch pose encoder + LoRA-LLM (`tsl.model`).

    Kept because it is the ablation for "does a pretrained sign encoder matter at
    all" — it shares no weights with any sign-language model, only with a text
    LLM.
    """

    arch = "scratch"

    def __init__(self, a):
        from tsl.model import ContextTranslator, ModelConfig

        d_in = {"pose": POSE_DIM, "dino": DINO_DIM, "rtmpose": RTMPOSE_DIM}[a.backend]
        self.d_in = d_in
        self._cfg = ModelConfig(d_in=d_in, llm=a.llm,
                                fusion="none" if a.system == "local" else a.fusion,
                                max_frames=a.max_frames)
        self.model = ContextTranslator(self._cfg).cuda()
        self.a = a

    def collate(self, batch):
        return collate(batch, self.d_in)

    def loss(self, batch, label_smoothing=None):
        return self.model(to_cuda(batch))

    def groups(self, a) -> list[dict]:
        named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        return [{"params": [p for n, p in named if "lora_" not in n],
                 "lr": a.lr_adapter, "name": "adapter"},
                {"params": [p for n, p in named if "lora_" in n],
                 "lr": a.lr_lora, "name": "lora"}]

    def cfg(self, a) -> dict:
        return dict(self._cfg.__dict__)

    def state(self) -> dict:
        return self.model.adapter_state()

    def extra(self) -> dict:
        gate = self.model.fuse.gate if self.model.fuse is not None else (0.0, 0.0)
        return {"gate_attn": gate[0], "gate_ff": gate[1]}


# ------------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--system", choices=["context", "local"], default="context")
    ap.add_argument("--arch", choices=["unisign", "scratch"], default="unisign")
    ap.add_argument("--init", default="csl_daily",
                    help=f"one of {', '.join(us.INITS)}, or a path to a .pth")
    ap.add_argument("--init-parts", default="all", choices=sorted(us.PART_SETS),
                    help="take only part of the checkpoint: `pose` is its sign "
                         "encoder on a vanilla mT5-base, `mt5` is its decoder on "
                         "a random encoder. Splits encoder credit from decoder "
                         "credit, which no whole-checkpoint comparison can. "
                         "`proj` / `pose_noproj` / `mt5_proj` further isolate "
                         "`pose_proj`, the linear map where the two halves meet.")
    ap.add_argument("--init-mt5", default="",
                    help="take the decoder from a SECOND checkpoint, so both "
                         "halves are pretrained but were never trained together. "
                         "With `--init X --init-parts pose --init-mt5 Y` the only "
                         "thing broken is the pairing, which is the one thing the "
                         "single-checkpoint split rows cannot test.")
    ap.add_argument("--init-mt5-parts", default="mt5", choices=sorted(us.PART_SETS),
                    help="which tensors to take from --init-mt5. Default `mt5` is "
                         "the whole text half. `lm_head` takes only the output "
                         "projection, which is where the released checkpoints "
                         "actually differ (scripts/21_lineage.py), so it asks "
                         "whether the decoder effect is that one tensor.")
    ap.add_argument("--tune", choices=["lora", "full"], default="lora")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--feat", type=Path, default=None)
    ap.add_argument("--backend", choices=["rtmpose", "pose", "dino"], default="rtmpose")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="--arch scratch only")
    ap.add_argument("--fusion", choices=["gated", "concat", "none"], default="gated",
                    help="--arch scratch only")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr-pose", type=float, default=3e-4,
                    help="Uni-Sign's own fine-tuning rate, applied to their pose "
                         "branch; identical across initialisations on purpose")
    ap.add_argument("--lr-lora", type=float, default=3e-4)
    ap.add_argument("--lr-mt5", type=float, default=3e-5, help="--tune full only")
    ap.add_argument("--lr-adapter", type=float, default=3e-4, help="--arch scratch only")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.2,
                    help="upstream's value, used for the training objective; the "
                         "dev loss is always plain cross-entropy")
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--max-frames", type=int, default=192)
    ap.add_argument("--ctx-max-frames", type=int, default=192)
    ap.add_argument("--ctx-k", type=int, default=4,
                    help="preceding utterances used as context; 4 reaches the "
                         "nearest antecedent for 75%% of anaphoric items")
    ap.add_argument("--amp", choices=["bf16", "off"], default="bf16")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="checkpoint the mT5; slower, needed for --tune full")
    ap.add_argument("--limit", type=int, default=0,
                    help="use only N train / N dev utterances (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--save-last", action="store_true",
                    help="also write last.pt at the final epoch, for the "
                         "model-selection sensitivity check (dev CE picks "
                         "best.pt, and dev CE is not the diagnostic)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    feat = a.feat or a.data / f"feat_{a.backend}"
    if a.arch == "unisign":
        init = a.init if a.init_parts == "all" else f"{a.init}-{a.init_parts}"
        if a.init_mt5:
            init = f"{init}+{a.init_mt5}-{a.init_mt5_parts}"
        name = a.tag or (f"f{a.fold}_{a.system}_{init}_{a.tune}"
                         f"_k{a.ctx_k}_s{a.seed}")
    else:
        name = a.tag or (f"f{a.fold}_{a.system}_{a.backend}_{a.fusion}"
                         f"_k{a.ctx_k}_s{a.seed}")
    outdir = a.out / name
    outdir.mkdir(parents=True, exist_ok=True)

    records = load_records(a.data / "records.jsonl")
    folds = load_folds(a.data / "folds.json")
    sp = split_records(records, folds, a.fold)
    donors = assign_donors(records, seed=a.seed)

    cond = "local" if a.system == "local" else TRAIN_MIX
    tr_recs, dev_recs = sp["train"], sp["dev"]
    if a.limit:
        tr_recs, dev_recs = tr_recs[: a.limit], dev_recs[: a.limit]
    ds_tr = TSLDataset(tr_recs, feat, condition=cond, max_frames=a.max_frames,
                       ctx_max_frames=a.ctx_max_frames, ctx_k=a.ctx_k, seed=a.seed,
                       donors=donors, all_records=records)
    ds_dev = TSLDataset(dev_recs, feat,
                        condition="local" if a.system == "local" else "both",
                        max_frames=a.max_frames, ctx_max_frames=a.ctx_max_frames,
                        ctx_k=a.ctx_k, seed=a.seed, donors=donors,
                        all_records=records)

    system = UniSignSystem(a) if a.arch == "unisign" else ScratchSystem(a)
    model = system.model

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=a.batch, shuffle=shuffle,
                          num_workers=a.workers, collate_fn=system.collate,
                          drop_last=shuffle, pin_memory=True,
                          persistent_workers=a.workers > 0)

    dl_tr, dl_dev = loader(ds_tr, True), loader(ds_dev, False)

    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    print(f"{name}: trainable {tr/1e6:.1f}M / {tot/1e6:.1f}M "
          f"({len(ds_tr)} train, {len(ds_dev)} dev, {len(dl_tr)} steps/epoch)")

    groups = [g for g in system.groups(a) if g["params"]]
    print("  " + "  ".join(f"{g['name']} {sum(p.numel() for p in g['params'])/1e6:.1f}M"
                           f"@{g['lr']:.0e}" for g in groups))
    opt = torch.optim.AdamW(groups, weight_decay=0.01, betas=(0.9, 0.95))
    steps = max(1, (len(dl_tr) // a.accum) * a.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=steps,
        pct_start=a.warmup, anneal_strategy="cos",
    )
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp == "bf16")
    params = [p for g in groups for p in g["params"]]

    best = float("inf")
    log = []
    t0 = time.time()
    step = 0
    for ep in range(a.epochs):
        model.train()
        run = 0.0
        for i, batch in enumerate(dl_tr):
            with amp:
                loss = system.loss(batch) / a.accum
            loss.backward()
            run += float(loss.detach()) * a.accum
            if (i + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if step < steps - 1:
                    sched.step()
                step += 1
        tr_loss = run / max(1, len(dl_tr))

        model.eval()
        dev = 0.0
        with torch.no_grad():
            for batch in dl_dev:
                with amp:
                    # Plain cross-entropy, never the smoothed objective: model
                    # selection should be on a likelihood, and the number has to
                    # stay comparable if the smoothing is ever changed.
                    dev += float(system.loss(batch, label_smoothing=0.0))
        dev /= max(1, len(dl_dev))

        mem = torch.cuda.max_memory_allocated() / 2**30
        extra = system.extra()
        note = "".join(f"  {k} {v:+.3f}" for k, v in extra.items())
        print(f"  ep{ep+1:02d} train {tr_loss:.4f}  dev {dev:.4f}{note}  "
              f"{mem:.1f}GB  {(time.time()-t0)/60:.1f}m", flush=True)
        log.append({"epoch": ep + 1, "train": tr_loss, "dev": dev, "gb": mem,
                    **extra})

        def _ckpt() -> dict:
            return {"arch": system.arch, "cfg": system.cfg(a),
                    "args": vars(a) | {"data": str(a.data), "feat": str(feat),
                                       "out": str(a.out)},
                    "state": system.state(), "epoch": ep + 1, "dev": dev}

        if dev < best:
            best = dev
            torch.save(_ckpt(), outdir / "best.pt")
        # ReviewerA's B6: model selection is on dev cross-entropy, and Appendix F
        # already shows dev CE is decoupled from the diagnostic (RAND-VIS has the
        # lowest dev CE and the second-worst accuracy). So the selection rule may
        # be costing different cells different amounts, and the check for that is
        # to re-read the same trajectory at its LAST epoch instead of its best.
        # Written only on request: this is a second 7.8 GB file per run.
        if a.save_last and ep + 1 == a.epochs:
            torch.save(_ckpt(), outdir / "last.pt")
        # Rewritten every epoch: the card is shared, and a run that is evicted
        # half way should still leave its curve behind.
        (outdir / "log.json").write_text(json.dumps(
            {"name": name, "arch": system.arch, "cfg": system.cfg(a),
             "log": log, "best_dev": best, "epochs_done": ep + 1,
             "minutes": (time.time() - t0) / 60,
             "peak_gb": torch.cuda.max_memory_allocated() / 2**30}, indent=2))

    print(f"best dev {best:.4f}  {(time.time()-t0)/60:.1f} min  -> {outdir}/best.pt")


def to_cuda(batch):
    if torch.is_tensor(batch):
        return batch.cuda(non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_cuda(v) for k, v in batch.items()}
    return batch


if __name__ == "__main__":
    main()
