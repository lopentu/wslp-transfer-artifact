"""Adapter around Uni-Sign: our features in, context-conditioned TSL→Chinese out.

Uni-Sign is the base model for the transfer axis. Its released checkpoints are
all 973.5 M parameters with identical architecture, which is what makes an
initialisation comparison fair; see `scripts/08_check_external.py`.

Their source is fetched, not vendored — the upstream repository has no licence.
Run `scripts/09_fetch_unisign.py` first. This module imports their `Uni_Sign`
model and their `load_part_kp`, and only their `forward` is replaced, because it
hardcodes a fixed prompt and has nowhere to put discourse context.

What we add
-----------
Uni-Sign encodes one clip. Its mT5 encoder input is
``[prefix text embeds] [pose embeds]``, so context has two natural entry points
that stay separable — which is the whole point of the modality ablation:

* **context text** extends the prefix,
* **context video** prepends a second block of pose embeds,

with the attention mask extended to match. Setting neither reproduces upstream
behaviour exactly, so `local` is a faithful baseline rather than a crippled one.

Normalisation is *theirs*, unchanged: wrist-relative hands, nose-relative face,
a body-derived scale, confidence gated at 0.3. We store raw COCO-WholeBody 133
keypoints precisely so their loader can do this untouched — reimplementing it by
eye would risk a silent numerical mismatch that would read as a bad transfer
result.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party" / "unisign"
MODES = ("body", "left", "right", "face_all")

# Which tensors of a released checkpoint to take, as a predicate over its keys.
# A checkpoint bundles an encoder that has seen sign language with a decoder
# narrowed to one written language, so no whole-checkpoint comparison can say
# which half a difference came from. These sets can.
#
# `pose_proj` is the 6.9 M pose branch's 0.8 M linear map into the decoder's
# embedding space — the one place where the two halves actually meet, and so
# the obvious seat of any co-adaptation between them. It is separable from the
# representation feeding it, which is what `proj` / `pose_noproj` / `mt5_proj`
# are for.
PART_SETS = {
    "all": lambda k: True,
    "pose": lambda k: not k.startswith("mt5_model"),
    "mt5": lambda k: k.startswith("mt5_model"),
    "proj": lambda k: k.startswith("pose_proj"),
    "pose_noproj": lambda k: not k.startswith("mt5_model")
    and not k.startswith("pose_proj"),
    "mt5_proj": lambda k: k.startswith("mt5_model") or k.startswith("pose_proj"),
    # Added 8/18, after measuring what actually differs between the released
    # checkpoints (`scripts/21_lineage.py`). Their mT5 encoder and decoder bodies
    # sit within half a percent of each other in relative L2, and their token
    # embeddings are identical to five significant figures; the divergence is
    # almost entirely in `lm_head`, whose cosine is 0.95 within the CSL pair and
    # 0.89 within the ASL pair but 0.28-0.31 across them. So "the decoder's
    # written language" has a candidate physical seat, and these two sets are how
    # a cross-load tests it: `lm_head` moves 192.1 M parameters of output
    # projection and nothing else, `mt5_nohead` moves everything but.
    "lm_head": lambda k: k.startswith("mt5_model.lm_head"),
    "mt5_nohead": lambda k: k.startswith("mt5_model")
    and not k.startswith("mt5_model.lm_head"),
}


def _import_models():
    """Import Uni-Sign's model definition from third_party."""
    if not (THIRD_PARTY / "models.py").exists():
        raise RuntimeError(
            f"Uni-Sign source not found at {THIRD_PARTY}.\n"
            "Run: python3 scripts/09_fetch_unisign.py"
        )
    p = str(THIRD_PARTY)
    if p not in sys.path:
        sys.path.insert(0, p)
    import models as us_models  # noqa: E402

    return us_models


# The pose normalisation we need lives in their datasets.py, but importing that
# module pulls in their utils.py and therefore deepspeed, an entire distributed
# training stack, for the sake of two pure-numpy functions. So lift just those two
# out of the source with `ast` and execute them in a namespace holding only what
# they use. This keeps the numerics byte-identical to theirs — which matters,
# because the released weights were trained under exactly this normalisation —
# without dragging in the training dependencies or copying their code into this
# repository.
_WANTED = ("crop_scale", "load_part_kp")
_upstream_fns: dict = {}


def _load_pose_normalisers() -> dict:
    if _upstream_fns:
        return _upstream_fns
    import ast
    import copy as _copy

    src_path = THIRD_PARTY / "datasets.py"
    if not src_path.exists():
        raise RuntimeError(f"{src_path} missing; run scripts/09_fetch_unisign.py")
    tree = ast.parse(src_path.read_text())
    wanted = [n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name in _WANTED]
    missing = set(_WANTED) - {n.name for n in wanted}
    if missing:
        raise RuntimeError(
            f"{sorted(missing)} not found in {src_path}. Upstream changed shape; "
            "re-check the normalisation before trusting any transfer result."
        )
    ns: dict = {"np": np, "torch": torch, "copy": _copy}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(src_path), "exec"), ns)
    _upstream_fns.update({k: ns[k] for k in _WANTED})
    return _upstream_fns


# ----------------------------------------------------------------- features


def parts_from_feature(feat: np.ndarray, force_ok: bool = False) -> dict:
    """Our stored (T, 133*3) -> the four normalised part tensors Uni-Sign wants.

    Delegates the actual normalisation to their `load_part_kp`, which expects the
    per-frame shapes their extractor produced: keypoints (1, 133, 2) and scores
    (1, 133). Ours are stored flat, so reshape and hand them over.
    """
    fns = _load_pose_normalisers()
    kp = np.asarray(feat, np.float32).reshape(len(feat), 133, 3)
    skeletons = [kp[t, :, :2][None] for t in range(len(kp))]
    confs = [kp[t, :, 2][None] for t in range(len(kp))]
    if not skeletons:
        return {m: torch.zeros(0, 1, 3) for m in MODES}
    return fns["load_part_kp"](skeletons, confs, force_ok=force_ok)


def collate_parts(batch_feats: list[np.ndarray], max_frames: int = 256) -> dict:
    """Pad a batch of clips into Uni-Sign's `src_input` dict."""
    per = []
    for f in batch_feats:
        if len(f) > max_frames:
            idx = np.linspace(0, len(f) - 1, max_frames).round().astype(int)
            f = f[idx]
        per.append(parts_from_feature(f))
    T = max(max(p["body"].shape[0] for p in per), 1)
    B = len(per)

    out: dict[str, torch.Tensor] = {}
    for m in MODES:
        V = per[0][m].shape[1]
        buf = torch.zeros(B, T, V, 3)
        for b, p in enumerate(per):
            n = p[m].shape[0]
            if n:
                buf[b, :n] = p[m].float()
        out[m] = buf
    mask = torch.zeros(B, T, dtype=torch.long)
    for b, p in enumerate(per):
        mask[b, : p["body"].shape[0]] = 1
    out["attention_mask"] = mask
    return out


N_KP = 133
FEAT_DIM = N_KP * 3


def _frames(x, dim: int = FEAT_DIM) -> np.ndarray:
    """(T, 399) float32, or one zero frame if the clip is absent or malformed.

    A zero frame is not a silent substitution: every keypoint confidence is 0, so
    `crop_scale` finds fewer than four valid points and returns zeros for every
    part. The row is then masked out by the caller. All 5,282 spans are on disk
    (`11_verify_features.py` is the gate), so this is a guard, not a code path
    that is expected to run.
    """
    if x is None:
        return np.zeros((1, dim), np.float32)
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != dim:
        return np.zeros((1, dim), np.float32)
    return x


META_KEYS = ("key", "condition", "tgt_src", "candidates", "item_type",
             "pron_gloss", "resolved", "prior_correct", "agreeing_verb",
             "ctx_dist", "n_candidates", "n_ctx", "tgt_ok")


def collate_unisign(items: list[dict], max_frames: int = 192,
                    ctx_max_frames: int = 192) -> dict:
    """`TSLDataset` items -> the arguments `UniSignTSL.forward` takes.

    The counterpart of `dataset.collate` for this model: that one pads flat
    frame vectors for the from-scratch encoder, this one hands each clip to
    Uni-Sign's own normalisation and packs the four part tensors.
    """
    src = collate_parts([_frames(it["tgt"]) for it in items], max_frames)

    have = torch.tensor([it.get("ctx") is not None and len(it["ctx"]) > 0
                         for it in items], dtype=torch.long)
    context_src = None
    if int(have.sum()):
        context_src = collate_parts([_frames(it.get("ctx")) for it in items],
                                    ctx_max_frames)
        # Mask out the rows that have no context, so a `local` example sitting in
        # a mixed batch is exactly `local` rather than "attends to a block of
        # zeros". Without this the context dropout schedule would be teaching the
        # model that absent context looks like a still signer.
        context_src["attention_mask"] = context_src["attention_mask"] * have[:, None]

    return {
        "src": src,
        "context_src": context_src,
        "context_text": [it.get("ctx_text") for it in items],
        "answer": [it["answer"] for it in items],
        "meta": [{k: it[k] for k in META_KEYS if k in it} for it in items],
    }


# -------------------------------------------------------------------- model


class UniSignTSL(nn.Module):
    """Uni-Sign with an optional discourse-context path."""

    def __init__(self, checkpoint: str | Path | None = None, lang: str = "Chinese",
                 hidden_dim: int = 256, label_smoothing: float = 0.2,
                 dataset: str = "CSL_Daily", max_target_len: int = 64,
                 parts: str = "all", checkpoint_mt5: str | Path | None = None,
                 parts_mt5: str = "mt5"):
        super().__init__()
        us_models = _import_models()
        args = SimpleNamespace(dataset=dataset, hidden_dim=hidden_dim,
                               label_smoothing=label_smoothing, rgb_support=False)
        self.core = us_models.Uni_Sign(args)
        # `lang` only decides the wording of the prompt upstream; our target is
        # Chinese whatever the checkpoint was pretrained on, so set it explicitly
        # rather than letting the source dataset name pick it.
        self.core.lang = lang
        self.tok = self.core.mt5_tokenizer
        self.mt5 = self.core.mt5_model
        self.max_target_len = max_target_len
        self.label_smoothing = label_smoothing
        self.lora = None
        self.loaded_from = None
        # Every load, in order. A second entry means the two halves came from
        # different checkpoints — the cross-pairing row, where both halves are
        # individually pretrained and only their pairing is broken.
        self.loads: list[dict] = []
        if checkpoint is not None:
            self.load_pretrained(checkpoint, parts=parts)
        if checkpoint_mt5 is not None:
            self.load_pretrained(checkpoint_mt5, parts=parts_mt5)

    # ------------------------------------------------------------ checkpoint

    def load_pretrained(self, path: str | Path, parts: str = "all") -> dict:
        """Load one of the released checkpoints; report what did not match.

        Reported rather than silently ignored: a transfer row that quietly failed
        to load its weights would look exactly like "transfer does not help".

        `parts` selects half of the checkpoint and leaves the other half at its
        published initialisation — the pose branch random, or the mT5 as
        `google/mt5-base` ships it:

            all    everything, i.e. the released model
            pose   Uni-Sign's pose branch only, on top of a vanilla mT5-base
            mt5    the fine-tuned mT5 only, on top of a random pose branch

        The released checkpoints bundle an encoder that has seen sign language
        with a decoder that has been narrowed to one written language, so no
        comparison between whole checkpoints can say which half a difference came
        from. These two rows can: `pose` is encoder credit with the decoder held
        constant, `mt5` is decoder credit with the encoder held constant.
        """
        try:
            keep = PART_SETS[parts]
        except KeyError:
            raise ValueError(f"parts must be one of {sorted(PART_SETS)}, "
                             f"got {parts!r}") from None
        obj = torch.load(path, map_location="cpu", weights_only=False)
        sd = obj
        for key in ("model", "state_dict", "module"):
            if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        n_file = len(sd)
        sd = {k: v for k, v in sd.items() if keep(k)}

        # Missing is measured against the half we asked for, not the whole model,
        # so a partial load reports 0 missing when it succeeded — otherwise every
        # partial row would trip the "did not match" guard in 04_train.py.
        core_keys = set(self.core.state_dict())
        target = {k for k in core_keys if keep(k)}
        missing = sorted(target - set(sd))
        unexpected = sorted(set(sd) - core_keys)
        self.core.load_state_dict(sd, strict=False)
        report = {
            "checkpoint": str(path),
            "parts": parts,
            "in_file": n_file,
            "loaded": len(sd) - len(unexpected),
            "missing": len(missing),
            "unexpected": len(unexpected),
            "missing_sample": missing[:5],
            "unexpected_sample": unexpected[:5],
        }
        self.loads.append(report)
        # Stays the *first* load, so a caller that knows nothing about
        # cross-pairing still sees the primary checkpoint. Callers that do
        # should read `self.loads`.
        self.loaded_from = self.loads[0]
        return report

    # ----------------------------------------------------------- pose branch

    def pose_embeds(self, src: dict) -> torch.Tensor:
        """(B, T, 768) — Uni-Sign's pose branch, run through their own modules.

        Mirrors the pose half of their `forward`: per-part projection, spatial
        GCN, the body-conditioned offsets for hands and face, temporal GCN, pool,
        concatenate, add `part_para`, project to the mT5 width.
        """
        core = self.core
        feats, body_feat = [], None
        for part in MODES:
            x = core.proj_linear[part](src[part]).permute(0, 3, 1, 2)  # B,C,T,V
            g = core.gcn_modules[part](x)
            if part == "body":
                body_feat = g
            elif part == "left":
                g = g + body_feat[..., -2][..., None].detach()
            elif part == "right":
                g = g + body_feat[..., -1][..., None].detach()
            elif part == "face_all":
                g = g + body_feat[..., 0][..., None].detach()
            g = core.fusion_gcn_modules[part](g)
            feats.append(g.mean(-1).transpose(1, 2))  # B,T,C
        emb = torch.cat(feats, dim=-1) + core.part_para
        return core.pose_proj(emb)

    # -------------------------------------------------------------- assembly

    def _prefix(self, n: int, context_text: list[str] | None, device) -> tuple:
        base = f"Translate sign language video to {self.core.lang}: "
        if context_text is None:
            texts = [base] * n
        else:
            texts = [(f"Context: {c} {base}" if c else base) for c in context_text]
        tok = self.tok(texts, padding="longest", truncation=True, max_length=192,
                       return_tensors="pt").to(device)
        return self.mt5.encoder.embed_tokens(tok["input_ids"]), tok["attention_mask"]

    def encoder_inputs(self, src: dict, context_src: dict | None = None,
                       context_text: list[str] | None = None):
        pose = self.pose_embeds(src)
        mask = src["attention_mask"]
        if context_src is not None:
            cpose = self.pose_embeds(context_src)
            pose = torch.cat([cpose, pose], dim=1)
            mask = torch.cat([context_src["attention_mask"], mask], dim=1)
        pre_emb, pre_mask = self._prefix(pose.shape[0], context_text, pose.device)
        return (torch.cat([pre_emb, pose], dim=1),
                torch.cat([pre_mask, mask], dim=1))

    # ----------------------------------------------------------------- heads

    def forward(self, src: dict, targets: list[str],
                context_src: dict | None = None,
                context_text: list[str] | None = None,
                label_smoothing: float | None = None) -> torch.Tensor:
        """Cross-entropy on the Chinese target.

        `label_smoothing` defaults to the value the model was built with, which
        is upstream's 0.2. Pass 0.0 to get plain cross-entropy — that is what the
        dev pass uses, so the number model selection is done on is a likelihood
        and not a smoothing-dependent quantity.
        """
        embeds, mask = self.encoder_inputs(src, context_src, context_text)
        lab = self.tok(targets, return_tensors="pt", padding=True, truncation=True,
                       max_length=self.max_target_len)["input_ids"]
        lab[lab == self.tok.pad_token_id] = -100
        lab = lab.to(embeds.device)
        out = self.mt5(inputs_embeds=embeds, attention_mask=mask, labels=lab,
                       return_dict=True)
        ls = self.label_smoothing if label_smoothing is None else label_smoothing
        if ls <= 0:
            return out.loss
        # Same objective as upstream's own training loop, which does not use the
        # model's built-in loss precisely because it wants the smoothing.
        return nn.functional.cross_entropy(
            out.logits.reshape(-1, out.logits.shape[-1]).float(), lab.reshape(-1),
            ignore_index=-100, label_smoothing=ls,
        )

    @torch.no_grad()
    def generate(self, src: dict, context_src: dict | None = None,
                 context_text: list[str] | None = None, num_beams: int = 4,
                 max_new_tokens: int = 64) -> list[str]:
        embeds, mask = self.encoder_inputs(src, context_src, context_text)
        ids = self.mt5.generate(inputs_embeds=embeds, attention_mask=mask,
                                num_beams=num_beams, max_new_tokens=max_new_tokens)
        return [s.strip() for s in self.tok.batch_decode(ids, skip_special_tokens=True)]

    @torch.no_grad()
    def score(self, src: dict, candidates: list[list[str]],
              context_src: dict | None = None,
              context_text: list[str] | None = None) -> list[list[dict]]:
        """Per-candidate log-likelihood, for contrastive evaluation."""
        embeds, mask = self.encoder_inputs(src, context_src, context_text)
        n = max(len(c) for c in candidates)
        out: list[list[dict]] = [[] for _ in candidates]
        for j in range(n):
            idx = [i for i, c in enumerate(candidates) if j < len(c)]
            if not idx:
                continue
            sel = torch.tensor(idx, device=embeds.device)
            lab = self.tok([candidates[i][j] for i in idx], return_tensors="pt",
                           padding=True, truncation=True,
                           max_length=self.max_target_len)["input_ids"].to(embeds.device)
            lab_masked = lab.clone()
            lab_masked[lab_masked == self.tok.pad_token_id] = -100
            logits = self.mt5(inputs_embeds=embeds[sel], attention_mask=mask[sel],
                              labels=lab_masked, return_dict=True).logits.float()
            lp = torch.log_softmax(logits, -1)
            keep = lab_masked != -100
            got = lp.gather(-1, lab_masked.clamp(min=0)[..., None])[..., 0] * keep
            tot, cnt = got.sum(-1), keep.sum(-1).clamp(min=1)
            for r, i in enumerate(idx):
                out[i].append({"sum": float(tot[r]), "mean": float(tot[r] / cnt[r])})
        return out

    # ------------------------------------------------------------ parameters

    def trainable_report(self) -> dict:
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"trainable_M": tr / 1e6,
                "total_M": sum(p.numel() for p in self.parameters()) / 1e6}

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def apply_lora(self, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        """LoRA on the mT5 attention/FFN; pose encoder and projection stay trainable.

        Needed because full fine-tuning is ~12.7 GB and the GPU on this machine is
        shared — see README §5.

        Must run *after* `load_pretrained`: it renames every mT5 key, so a state
        dict from a released checkpoint would no longer match anything.
        """
        from peft import LoraConfig, get_peft_model

        for p in self.mt5.parameters():
            p.requires_grad = False
        self.core.mt5_model = get_peft_model(
            self.mt5,
            LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                       target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
                       task_type="SEQ_2_SEQ_LM"),
        )
        self.mt5 = self.core.mt5_model
        self.lora = {"r": r, "alpha": alpha, "dropout": dropout}
        return self.trainable_report()

    # ------------------------------------------------------------- checkpointing

    #: Everything outside the mT5 — Uni-Sign's pose branch. Prefix match, so this
    #: picks up the BatchNorm running statistics inside the ST-GCN blocks as well
    #: as the weights. Those are buffers, not parameters: selecting by
    #: `requires_grad` alone would train them and then not save them, and the
    #: checkpoint would evaluate with mT5's pretrained stats over our pose
    #: distribution.
    POSE_PREFIXES = ("core.proj_linear", "core.gcn_modules",
                     "core.fusion_gcn_modules", "core.part_para", "core.pose_proj")

    def trainable_state(self) -> dict:
        """What this run actually changed — small when LoRA is on, whole when not.

        Under LoRA the frozen mT5 is not stored, so reloading needs the *same*
        initialisation checkpoint underneath; `load_run` handles that, and the
        run's `cfg` records which one it was.
        """
        if self.lora is None:  # full fine-tune: nothing is recoverable from elsewhere
            return {k: v.detach().cpu() for k, v in self.state_dict().items()}
        return {k: v.detach().cpu() for k, v in self.state_dict().items()
                if k.startswith(self.POSE_PREFIXES) or "lora_" in k}

    def load_trainable_state(self, sd: dict) -> None:
        out = self.load_state_dict(sd, strict=False)
        if out.unexpected_keys:
            raise RuntimeError(
                f"{len(out.unexpected_keys)} unexpected keys, e.g. "
                f"{out.unexpected_keys[:3]} — the run was probably trained with a "
                "different --tune setting than this rebuild."
            )
        expected = {k for k in self.state_dict()
                    if k.startswith(self.POSE_PREFIXES) or "lora_" in k}
        gap = expected - set(sd)
        if gap:
            raise RuntimeError(f"{len(gap)} trained tensors absent from the "
                               f"checkpoint, e.g. {sorted(gap)[:3]}")


# ------------------------------------------------------------- initialisations

# The transfer axis (README §4). All five released checkpoints are 973.5 M /
# 627 tensors, so a row-to-row difference is a difference in what the weights saw
# and not in capacity. `random` is the control: same architecture, same
# mT5-base, only the 6.9 M pose branch left untrained — it is what "no
# sign-language pretraining" means here, not an untrained decoder.
INITS = {
    "random": None,
    "csl_daily": "csl_daily_pose_only_slt.pth",   # CSL  -> Chinese
    "csl_stage1": "csl_stage1_weight.pth",        # CSL-News pretrain
    "how2sign": "how2sign_pose_only_slt.pth",     # ASL  -> English
    "openasl": "openasl_pose_only_slt.pth",       # ASL  -> English
    "wlasl": "wlasl_pose_only_islr.pth",          # ASL isolated signs
    # Not pretrained models, and not new arms of the transfer axis: both are the
    # published mT5-base written to a file so that *parts* of it become
    # addressable by `--init-mt5-parts`, which is the only way to give a cell a
    # THIRD level of the output-projection factor (scripts/22_mt5base_ckpt.py,
    # scripts/23_text_lm.py). `random` already uses mt5_base's weights whole; the
    # difference is that these can be cross-loaded one tensor at a time.
    "mt5_base": "mt5_base.pth",                   # mT5-base, no sign-language FT
    "tsl_text": "tsl_text_lm.pth",                # mT5-base + TSL Chinese text only
    # The 8/22 reviews' controls on the neutral-head result, all of them levels of
    # the output-projection factor that no released file carries. Registered by
    # NAME rather than passed as a path on purpose: 04_train.py builds a run's
    # directory name out of `--init-mt5`, and 19_bootstrap.py cross-checks that
    # name against the cfg, so a bare path would either produce a directory with
    # a slash in it or a cell the analysis refuses to label.
    #
    # scripts/27_head_donors.py writes all five. `rand_head` is transformers' own
    # init for this tensor, which answers ReviewerO's P3 (is the rescue a RESET,
    # or is mT5-base's geometry special?) but sits at 2.2x mT5-base's row norm;
    # `rand_head_nm` is the same directions at mT5-base's per-row norms, which is
    # the version of P3 with the scale confound removed. The three `how2sign_*`
    # donors answer ReviewerA's W2/A3 (is the ASL head's damage only a per-script
    # logit SCALE?): `cjk` restores the CJK:Latin norm ratio to mT5-base's 1.02,
    # `gl` restores the overall radius, `cjkgl` both.
    "rand_head": "head_donors/rand_head.pth",
    "rand_head_nm": "head_donors/rand_head_nm.pth",
    # Camera-ready reviewer 2's control on the neutral-head rescue. Every other
    # donor changes the matrix's contents and its token correspondence at once,
    # so neither can be credited alone. These two keep mT5-base's rows exactly --
    # same Frobenius norm, same row-norm distribution, same singular values --
    # and move only which token each row scores (scripts/27_head_donors.py).
    # `_perm` permutes the whole vocabulary, `_permws` within CJK/Latin/other.
    "mt5_base_perm": "head_donors/mt5_base_perm.pth",
    "mt5_base_permws": "head_donors/mt5_base_permws.pth",
    "how2sign_cjk": "head_donors/how2sign_cjk1p6059.pth",
    "how2sign_gl": "head_donors/how2sign_gl.pth",
    "how2sign_cjkgl": "head_donors/how2sign_cjkgl.pth",
    # ReviewerA's B2: mT5-base fine-tuned gloss->Chinese, the same
    # seq2seq-into-Chinese objective the sign checkpoints had with the video
    # removed (scripts/28_gloss_lm.py). Unlike `tsl_text` this is a task whose
    # loss does not start near zero, so it is the strong version of the
    # written-language instrument Appendix O only approximates.
    "gloss_lm": "gloss_lm.pth",
    # ReviewerG asks for a stronger text-only Chinese instrument and ReviewerA
    # calls the existing one weak; both are right, and the reason Appendix O gives
    # is the objective rather than the corpus (mT5 already models Chinese, so
    # span-denoising gradients are small and `lm_head` moves 0.026 against
    # 0.92-0.98 for a sign fine-tune). No larger Chinese corpus is on this box, so
    # the fix is applied to the movement instead: same corpus and objective at a
    # higher learning rate for longer, run until the projection has travelled as
    # far as a released sign fine-tune's did. Then a null from it is a null from a
    # comparable instrument, which is what the objection is actually about.
    "tsl_text_strong": "tsl_text_lm_strong.pth",
}


def init_path(init: str, external: str | Path = "data/external") -> Path | None:
    """Resolve an `--init` name (or an explicit path) to a checkpoint file."""
    if init in INITS:
        name = INITS[init]
        if name is None:
            return None
        p = Path(external) / name
    else:
        p = Path(init)
    if not p.exists():
        raise SystemExit(f"checkpoint {p} not found; run scripts/08_check_external.py "
                         f"--download {p.name}")
    return p


def build(cfg: dict) -> "UniSignTSL":
    """Rebuild the model a run was trained with, weights aside.

    Stated once and used by both training and evaluation, so an evaluation
    cannot quietly differ from the thing it is evaluating.
    """
    model = UniSignTSL(checkpoint=cfg.get("checkpoint") or None,
                       lang=cfg.get("lang", "Chinese"),
                       label_smoothing=cfg.get("label_smoothing", 0.2),
                       max_target_len=cfg.get("max_target_len", 64),
                       parts=cfg.get("init_parts", "all"),
                       checkpoint_mt5=cfg.get("checkpoint_mt5") or None,
                       parts_mt5=cfg.get("init_mt5_parts", "mt5"))
    if cfg.get("tune", "lora") == "lora":
        model.apply_lora(r=cfg.get("lora_r", 16), alpha=cfg.get("lora_alpha", 32),
                         dropout=cfg.get("lora_dropout", 0.05))
    return model


def load_run(path: str | Path, map_location="cpu") -> tuple["UniSignTSL", dict]:
    """A trained run from `runs/<name>/best.pt` -> (model, checkpoint dict).

    Under LoRA the frozen mT5 is not in the file, so the initialisation
    checkpoint recorded in `cfg` is loaded first and the trained tensors go on
    top. If that file has moved, this raises rather than evaluating a model that
    is silently half pretrained.
    """
    # mmap=True: `best.pt` is 7.8 GB (the state dict stores the mT5's shared
    # embedding under four keys, as the parameter-count note in the paper
    # explains), and a plain load materialises all of it in host RAM. Evaluation
    # then runs beside a training job on a 31 GB box that also has fourteen other
    # users on it, and on 8/18 the kernel OOM-killer took three evaluation cells
    # for exactly this reason. Memory-mapping pages the tensors in as they are
    # copied into the model instead.
    ck = torch.load(path, map_location=map_location, weights_only=False, mmap=True)
    if ck.get("arch") != "unisign":
        raise SystemExit(f"{path} was trained with arch={ck.get('arch')!r}, not unisign")
    model = build(ck["cfg"])
    model.load_trainable_state(ck["state"])
    return model, ck


# ------------------------------------------------------------------ selftest


def selftest(checkpoint: str | None = None, feat_dir: str = "data/feat_rtmpose"):
    """Build the model on real features and check every shape lines up."""
    import glob

    files = sorted(glob.glob(f"{feat_dir}/*.npz"))
    if not files:
        raise SystemExit(f"no features in {feat_dir}; run 03_extract_features.py")
    z = np.load(files[0])
    keys = sorted(z.files)[:2]
    feats = [z[k].astype(np.float32) for k in keys]
    print(f"clips: {keys} shapes {[f.shape for f in feats]}")

    src = collate_parts(feats)
    for m in MODES:
        print(f"  {m:9s} {tuple(src[m].shape)}")
    print(f"  attention_mask {tuple(src['attention_mask'].shape)}")

    model = UniSignTSL(checkpoint=checkpoint)
    if model.loaded_from:
        print(f"  checkpoint: {model.loaded_from}")
    print(f"  params: {model.trainable_report()}")

    with torch.no_grad():
        emb = model.pose_embeds(src)
        print(f"  pose_embeds {tuple(emb.shape)}  (expect (2, T, 768))")
        e, m_ = model.encoder_inputs(src, context_src=src,
                                     context_text=["前文一", "前文二"])
        print(f"  encoder_inputs with context: {tuple(e.shape)} mask {tuple(m_.shape)}")
        loss = model(src, ["測試句子一", "測試句子二"])
        print(f"  loss {float(loss):.4f}")
    print("OK")


if __name__ == "__main__":
    selftest(*sys.argv[1:])
