"""Context-conditioned TSL -> Chinese translator.

    frozen visual features
        -> sign encoder (Transformer, trained)
        -> perceiver resampler to a fixed number of visual tokens (trained)
        -> optional gated fusion with the context clip's tokens (trained)
        -> projection into the LLM embedding space (trained)
        -> causal LM with LoRA (base frozen)

Design notes that matter for the claim:

* The context gate is a Flamingo-style ``tanh(alpha)`` with ``alpha`` initialised
  to 0, so at step 0 the context branch is exactly a no-op. The context can only
  ever help by being *learned* to help, and ``alpha`` is itself reportable — a
  model that ignores visual context leaves it near zero.
* Visual context enters through the gate; textual context enters through the
  prompt. Keeping the two paths separate is what lets the modality ablation
  attribute a gain to one or the other (plan §6.2).
* The decoder is small on purpose. Sign2GPT found 564M within 0.23 BLEU-4 of
  1.7B, and a small decoder is what makes a full condition sweep affordable
  before the deadline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    d_in: int = 228  # tsl.features.POSE_DIM
    d_model: int = 512
    n_layers: int = 4
    n_heads: int = 8
    n_visual_tokens: int = 16
    dropout: float = 0.1
    llm: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )
    fusion: str = "gated"  # 'none' | 'concat' | 'gated'
    max_frames: int = 256
    max_text: int = 160


def sinusoids(t: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(t, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)[None]
    ang = pos / torch.pow(10000.0, i / d)
    return torch.cat([ang.sin(), ang.cos()], -1)[:, :d]


class SignEncoder(nn.Module):
    """Frame features -> a fixed number of visual tokens."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        self.inp = nn.Sequential(
            nn.LayerNorm(c.d_in), nn.Linear(c.d_in, c.d_model), nn.GELU(),
            nn.Linear(c.d_model, c.d_model),
        )
        layer = nn.TransformerEncoderLayer(
            c.d_model, c.n_heads, c.d_model * 4, c.dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.body = nn.TransformerEncoder(layer, c.n_layers)
        self.query = nn.Parameter(torch.randn(c.n_visual_tokens, c.d_model) * 0.02)
        self.pool = nn.MultiheadAttention(c.d_model, c.n_heads, dropout=c.dropout,
                                          batch_first=True)
        self.out = nn.LayerNorm(c.d_model)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x (B,T,d_in); valid (B,T) True where the frame is real."""
        h = self.inp(x)
        h = h + sinusoids(h.shape[1], h.shape[2], h.device)[None].to(h.dtype)
        pad = ~valid
        # A row that is entirely padding makes softmax produce NaN; keep frame 0.
        pad = pad & ~(pad.all(1, keepdim=True) & (torch.arange(
            pad.shape[1], device=pad.device)[None] == 0))
        h = self.body(h, src_key_padding_mask=pad)
        q = self.query[None].expand(h.shape[0], -1, -1).to(h.dtype)
        z, _ = self.pool(q, h, h, key_padding_mask=pad, need_weights=False)
        return self.out(z)


class GatedFusion(nn.Module):
    """Target visual tokens attend to context visual tokens, zero-initialised."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(c.d_model, c.n_heads, dropout=c.dropout,
                                          batch_first=True)
        self.norm_q = nn.LayerNorm(c.d_model)
        self.norm_kv = nn.LayerNorm(c.d_model)
        self.ff = nn.Sequential(
            nn.LayerNorm(c.d_model), nn.Linear(c.d_model, c.d_model * 4), nn.GELU(),
            nn.Linear(c.d_model * 4, c.d_model),
        )
        self.alpha_attn = nn.Parameter(torch.zeros(1))
        self.alpha_ff = nn.Parameter(torch.zeros(1))

    def forward(self, tgt: torch.Tensor, ctx: torch.Tensor,
                has_ctx: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(self.norm_q(tgt), self.norm_kv(ctx), self.norm_kv(ctx),
                         need_weights=False)
        keep = has_ctx.to(tgt.dtype)[:, None, None]
        tgt = tgt + torch.tanh(self.alpha_attn) * a * keep
        return tgt + torch.tanh(self.alpha_ff) * self.ff(tgt)

    @property
    def gate(self) -> tuple[float, float]:
        return (float(torch.tanh(self.alpha_attn)), float(torch.tanh(self.alpha_ff)))


class ContextTranslator(nn.Module):
    def __init__(self, c: ModelConfig, dtype=torch.bfloat16):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.c = c
        self.tok = AutoTokenizer.from_pretrained(c.llm)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(c.llm, dtype=dtype)
        base.config.use_cache = False  # re-enabled inside generate()
        self.llm_dtype = dtype  # explicit: PeftModel.dtype resolves by delegation
        self.llm = get_peft_model(
            base,
            LoraConfig(r=c.lora_r, lora_alpha=c.lora_alpha, lora_dropout=c.lora_dropout,
                       target_modules=list(c.lora_targets), task_type="CAUSAL_LM"),
        )
        d_llm = base.config.hidden_size

        self.enc = SignEncoder(c)
        self.fuse = GatedFusion(c) if c.fusion == "gated" else None
        self.proj = nn.Sequential(nn.LayerNorm(c.d_model), nn.Linear(c.d_model, d_llm))
        # Adapter stack in fp32 for stable optimisation; only the LLM is bf16.
        self.enc.float()
        self.proj.float()
        if self.fuse is not None:
            self.fuse.float()

    # ------------------------------------------------------------------ pieces

    def visual_tokens(self, batch: dict) -> torch.Tensor:
        """(B, K, d_llm) — the prefix handed to the LLM."""
        z = self.enc(batch["tgt_feat"].float(), batch["tgt_valid"])
        ctx = batch.get("ctx_feat")
        if ctx is not None and self.c.fusion != "none":
            zc = self.enc(ctx.float(), batch["ctx_valid"])
            if self.c.fusion == "gated":
                z = self.fuse(z, zc, batch["has_ctx"])
            else:  # concat: context tokens simply precede the target's
                zc = zc * batch["has_ctx"].float()[:, None, None]
                z = torch.cat([zc, z], 1)
        return self.proj(z)

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(ids)

    def _assemble(self, batch: dict, answers: list[str]):
        """[visual prefix][prompt][answer] with loss only on [answer]."""
        vis = self.visual_tokens(batch)
        B, K, _ = vis.shape
        dev = vis.device
        prompts = batch["prompt"]
        rows = []
        for i in range(B):
            p = self.tok(prompts[i], add_special_tokens=False).input_ids
            a = self.tok(answers[i], add_special_tokens=False).input_ids
            a = a[: self.c.max_text] + [self.tok.eos_token_id]
            rows.append((p, a))
        width = K + max(len(p) + len(a) for p, a in rows)

        embeds = torch.zeros(B, width, vis.shape[-1], dtype=vis.dtype, device=dev)
        attn = torch.zeros(B, width, dtype=torch.long, device=dev)
        labels = torch.full((B, width), -100, dtype=torch.long, device=dev)
        for i, (p, a) in enumerate(rows):
            ids = torch.tensor(p + a, device=dev)
            n = K + len(ids)
            embeds[i, :K] = vis[i]
            embeds[i, K:n] = self._embed(ids).to(vis.dtype)
            attn[i, :n] = 1
            labels[i, K + len(p) : n] = torch.tensor(a, device=dev)
        return embeds, attn, labels

    # ------------------------------------------------------------------- heads

    def forward(self, batch: dict) -> torch.Tensor:
        embeds, attn, labels = self._assemble(batch, batch["answer"])
        out = self.llm(inputs_embeds=embeds.to(self.llm_dtype), attention_mask=attn,
                       labels=labels)
        return out.loss

    @torch.no_grad()
    def score(self, batch: dict, candidates: list[list[str]]) -> list[list[dict]]:
        """Log-likelihood of each candidate under the same visual prefix.

        Returns per example a list of {sum, mean} log-probs. Minimal pairs differ
        in length (他 vs 他們), so the caller picks: `mean` removes the length
        prior, `sum` is the usual contrastive-pair convention. Both are reported.
        """
        n = max(len(c) for c in candidates)
        out: list[list[dict]] = [[] for _ in candidates]
        for j in range(n):
            idx = [i for i, c in enumerate(candidates) if j < len(c)]
            if not idx:
                continue
            sub = {k: (v[idx] if torch.is_tensor(v) else [v[i] for i in idx])
                   for k, v in batch.items()}
            embeds, attn, labels = self._assemble(sub, [candidates[i][j] for i in idx])
            logits = self.llm(inputs_embeds=embeds.to(self.llm_dtype),
                              attention_mask=attn).logits.float()
            lp = torch.log_softmax(logits[:, :-1], -1)
            tgt = labels[:, 1:]
            keep = tgt != -100
            picked = lp.gather(-1, tgt.clamp(min=0)[..., None])[..., 0] * keep
            tot = picked.sum(-1)
            cnt = keep.sum(-1).clamp(min=1)
            for r, i in enumerate(idx):
                out[i].append({"sum": float(tot[r]), "mean": float(tot[r] / cnt[r])})
        return out

    @torch.no_grad()
    def generate(self, batch: dict, max_new_tokens: int = 64) -> list[str]:
        vis = self.visual_tokens(batch)
        B, K, _ = vis.shape
        dev = vis.device
        ids = [self.tok(p, add_special_tokens=False).input_ids for p in batch["prompt"]]
        width = K + max(len(p) for p in ids)
        embeds = torch.zeros(B, width, vis.shape[-1], dtype=vis.dtype, device=dev)
        attn = torch.zeros(B, width, dtype=torch.long, device=dev)
        for i, p in enumerate(ids):  # left-pad so every row ends at the last token
            n = K + len(p)
            off = width - n
            embeds[i, off : off + K] = vis[i]
            embeds[i, off + K :] = self._embed(torch.tensor(p, device=dev)).to(vis.dtype)
            attn[i, off:] = 1
        seq = self.llm.generate(
            inputs_embeds=embeds.to(self.llm_dtype), attention_mask=attn,
            max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            use_cache=True,
            pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id,
        )
        # With inputs_embeds and no input_ids, generate() returns ONLY the new
        # tokens, so there is no prompt prefix to strip here.
        return [self.tok.decode(s, skip_special_tokens=True).strip() for s in seq]

    # ------------------------------------------------------------- checkpointing

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def n_trainable(self) -> tuple[int, int]:
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return tr, sum(p.numel() for p in self.parameters())

    def adapter_state(self) -> dict:
        """Everything trained: the adapters plus the LoRA deltas."""
        keep = {}
        for k, v in self.state_dict().items():
            if k.startswith(("enc.", "proj.", "fuse.")) or "lora_" in k:
                keep[k] = v.detach().cpu()
        return keep

    def load_adapter_state(self, sd: dict) -> None:
        missing = self.load_state_dict(sd, strict=False)
        unexpected = [k for k in missing.unexpected_keys]
        if unexpected:
            raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:5]}")
