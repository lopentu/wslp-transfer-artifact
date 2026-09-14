#!/usr/bin/env python3
"""Can an interface map rescue what the post-hoc graft could not?

The paper's §3.3 null -- graft CSL-Daily's fine-tuned `lm_head` onto a fine-tuned
How2Sign model and nothing happens (-1.5 points) -- is read as evidence that a
mismatched output projection prevented a usable visual representation from
FORMING, rather than hiding one that is there. ReviewerA's W3 is that the null
does not license that reading yet, and the objection is exactly right:

    the graft installs CSL-Daily's projection onto a decoder body that never
    co-adapted with it. The two models do not share a coordinate frame, so a
    failure to rescue is equally consistent with "there is nothing to read" and
    with "there is something to read, in the wrong basis".

Appendix M's cosine 0.991 does not answer this. That is the similarity of two
PROJECTIONS; the objection is about where the recipient's HIDDEN STATES live.

Model stitching (Lenc & Vedaldi 2015; Bansal et al. 2021) -- which ReviewerA also
notes is the missing citation here -- says how to settle it: put a linear map at
the interface, fit it, and see whether the two halves can be made to talk. If a
fitted interface still cannot rescue, "wrong basis" is dead and the trajectory
account survives. If it can, the paper's §3.3 conclusion has to be rewritten,
and we would rather find that out than have a reviewer find it.

Three maps, in increasing order of how much they are allowed to do -- the point
of the ladder is that a null at the TOP of it is a strong result:

  procrustes  the orthogonal M minimising ||H_r M - H_d||, i.e. rotation only.
              The weakest claim: the two frames differ by a rigid motion.
  ridge       the unconstrained linear M, L2-regularised. Allows rescaling and
              shearing between the frames.
  learned     M fitted by gradient descent to MINIMISE THE DONOR HEAD'S OWN
              CROSS-ENTROPY on the recipient's hidden states. This does not
              imitate the donor's states at all -- it directly optimises the
              thing a rescue would need, with 0.6M free parameters against the
              192.1M the graft moves. If even this does not rescue, no readout
              account survives.

Every map is folded into the tensor rather than applied at run time, which is
what makes this comparable to the graft it is testing: `lm_head` has no bias and
logits = h W^T, so mapping h -> M h is exactly the same model as replacing W with
W M. The output is therefore an ordinary synthetic donor and is scored by the
same `05_eval.py --posthoc-head` path as every other row in that table.

Fitted on the fold's DEV split only. The diagnostic items live in test, and a map
fitted on them would be a different experiment with a much better number.

    ../.venv/bin/python scripts/29_interface_align.py \
        --recipient f0_local_how2sign_full_k4_s0 \
        --donor f0_local_csl_daily_full_k4_s0

-> data/external/head_donors/align_<method>_<recipient>__<donor>.run.pth, plus a
fit report in data/interface_align.json. Forward passes and a 768x768 solve;
`--method learned` adds a few hundred steps on 0.6M parameters.
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
DONORS = ROOT / "data" / "external" / "head_donors"
RUN_KEY = "core.mt5_model.lm_head.weight"


def to_cuda(b):
    if torch.is_tensor(b):
        return b.cuda()
    if isinstance(b, dict):
        return {k: to_cuda(v) for k, v in b.items() if k != "meta"}
    return b


class HeadTap:
    """Capture what `lm_head` is actually fed.

    A pre-hook rather than `output_hidden_states`: T5 rescales the decoder stack's
    output by d_model**-0.5 on some configurations and not others, and the
    quantity this script needs is whatever the projection sees, not whatever the
    last block emitted.
    """

    def __init__(self, model):
        self.buf: list[torch.Tensor] = []
        self.h = model.mt5.lm_head.register_forward_pre_hook(self._hook)

    def _hook(self, _mod, inp):
        self.buf.append(inp[0].detach())

    def take(self) -> torch.Tensor:
        x = self.buf[-1]
        self.buf.clear()
        return x

    def close(self):
        self.h.remove()


@torch.no_grad()
def collect(model, loader, tok, max_len: int, want_states: bool = True):
    """Teacher-forced hidden states at `lm_head`, with the label mask.

    Teacher forcing is what makes two models comparable position by position: the
    decoder input is the gold prefix in both, so row i of one model's states and
    row i of the other's describe the same target position of the same utterance.
    Free-running decoding would diverge after the first disagreement and the
    correspondence the fit needs would not exist.
    """
    tap = HeadTap(model) if want_states else None
    H, Y, U = [], [], []
    uid = 0
    for batch in loader:
        b = to_cuda(batch)
        embeds, mask = model.encoder_inputs(b["src"], None, None)
        lab = tok(batch["answer"], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len)["input_ids"]
        lab[lab == tok.pad_token_id] = -100
        lab = lab.to(embeds.device)
        model.mt5(inputs_embeds=embeds, attention_mask=mask, labels=lab,
                  return_dict=True)
        h = tap.take()
        keep = lab != -100
        H.append(h[keep].float().cpu())
        Y.append(lab[keep].cpu())
        # Which utterance each surviving position came from, so the fit/holdout
        # split can be made at the utterance level. A token-level split would put
        # some of every utterance in both halves and the reported fit quality
        # would be partly in-sample.
        rows = torch.arange(lab.shape[0], device=lab.device)[:, None]
        U.append(rows.expand_as(lab)[keep].cpu() + uid)
        uid += lab.shape[0]
    if tap is not None:
        tap.close()
    return torch.cat(H), torch.cat(Y), torch.cat(U)


# ------------------------------------------------------------------------ maps


def fit_procrustes(hr: torch.Tensor, hd: torch.Tensor) -> torch.Tensor:
    """Orthogonal M with H_r M ~ H_d. No translation: a bias cannot be folded
    into a bias-free projection, so allowing one here would make the resulting
    donor a different model from the one that gets scored."""
    m = hr.double().T @ hd.double()
    u, _, vt = torch.linalg.svd(m, full_matrices=False)
    return (u @ vt).float()


def fit_ridge(hr: torch.Tensor, hd: torch.Tensor, lam: float) -> torch.Tensor:
    x = hr.double()
    a = x.T @ x
    a += lam * torch.eye(a.shape[0], dtype=a.dtype) * float(a.diagonal().mean())
    return torch.linalg.solve(a, x.T @ hd.double()).float()


def fit_learned(hr: torch.Tensor, y: torch.Tensor, w_donor: torch.Tensor,
                steps: int, lr: float, batch: int, seed: int = 0) -> tuple:
    """M minimising CE(y | (H_r M) W_donor^T), W_donor frozen.

    The most generous readout hypothesis that can be stated: the donor's 192.1M
    projection is fixed, and 0.6M interface parameters are free to make the
    recipient's states fit it as well as they possibly can, on data the recipient
    was trained on. A null here is the end of the readout account.
    """
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w = w_donor.to(dev).float()
    m = torch.eye(hr.shape[1], device=dev).requires_grad_(True)
    opt = torch.optim.Adam([m], lr=lr)
    n = hr.shape[0]
    log = []
    for step in range(steps):
        idx = torch.randint(0, n, (batch,))
        h = hr[idx].to(dev)
        t = y[idx].to(dev)
        logits = (h @ m) @ w.T
        loss = torch.nn.functional.cross_entropy(logits.float(), t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            log.append({"step": step, "ce": float(loss)})
            print(f"    step {step:5d}  CE {float(loss):.4f}", flush=True)
    return m.detach().cpu(), log


@torch.no_grad()
def teacher_ce(h: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
               m: torch.Tensor | None = None, chunk: int = 1024) -> float:
    """Cross-entropy of `y` under head `w` (optionally after interface `m`).

    Chunked because the full matrix is n x 250112 and the dev split has a few
    thousand positions.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w = w.to(dev).float()
    mm = None if m is None else m.to(dev).float()
    tot = cnt = 0.0
    for i in range(0, h.shape[0], chunk):
        x = h[i:i + chunk].to(dev).float()
        if mm is not None:
            x = x @ mm
        lg = x @ w.T
        tot += float(torch.nn.functional.cross_entropy(
            lg, y[i:i + chunk].to(dev), reduction="sum"))
        cnt += x.shape[0]
    return tot / max(cnt, 1)


def head_of_any(ckpt: Path) -> torch.Tensor:
    """`lm_head` out of a run checkpoint or a synthetic donor — both use
    `state`/`core.mt5_model.lm_head.weight`, so one reader covers both."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    sd = ck.get("state", ck)
    for k, v in sd.items():
        if k == RUN_KEY:
            return v.detach().float().clone()
    raise SystemExit(f"{ckpt} has no {RUN_KEY}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipient", required=True)
    ap.add_argument("--donor", default=None,
                    help="a run whose fine-tuned head is the donor. Its hidden "
                         "states are collected too, which is what procrustes and "
                         "ridge need.")
    ap.add_argument("--donor-head", default=None,
                    help="a synthetic donor (data/external/head_donors/*.run.pth) "
                         "to use instead of a run. `learned` only: with no donor "
                         "MODEL there are no donor hidden states to imitate, and "
                         "the imitation maps are undefined. This is how the "
                         "neutral mT5-base projection -- which is not a run -- "
                         "gets the same test as the CSL-Daily one.")
    ap.add_argument("--methods", default="procrustes,ridge,learned")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--folds", type=Path, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--ridge-lam", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fit-batch", type=int, default=512,
                    help="a logit tensor is batch x 250112; 512 keeps the\n                         step under ~1.5 GB so this can run beside a\n                         training cell")
    ap.add_argument("--holdout", type=float, default=0.2,
                    help="fraction of the dev positions held out of the fit, so "
                         "the reported fit quality is not in-sample")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "interface_align.json")
    a = ap.parse_args()

    records = load_records(a.data / "records.jsonl")
    folds = load_folds(a.folds or a.data / "folds.json")

    def build(name: str):
        ckpt = ROOT / "runs" / name / "best.pt"
        if not ckpt.exists():
            raise SystemExit(f"{name} has no best.pt")
        ck = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
        model, _ = us.load_run(ckpt)
        return model.cuda().eval(), ck, ckpt

    rec_model, rec_ck, rec_path = build(a.recipient)
    fold = rec_ck["args"]["fold"]
    cfg = rec_ck["cfg"]
    feat = Path(rec_ck["args"]["feat"])
    seed = rec_ck["args"].get("seed", 0)
    ctx_k = rec_ck["args"].get("ctx_k", 4)
    maxlen = cfg.get("max_target_len", 64)
    dev_recs = split_records(records, folds, fold)["dev"]
    donors = assign_donors(records, seed=seed)
    print(f"fold {fold}: {len(dev_recs)} dev utterances")

    def loader():
        ds = TSLDataset(dev_recs, feat, condition="local", ctx_k=ctx_k,
                        donors=donors, all_records=records,
                        max_frames=cfg.get("max_frames", 192),
                        ctx_max_frames=cfg.get("ctx_max_frames", 192))
        return DataLoader(ds, batch_size=a.batch, shuffle=False,
                          num_workers=a.workers,
                          collate_fn=lambda b: us.collate_unisign(
                              b, cfg.get("max_frames", 192),
                              cfg.get("ctx_max_frames", 192)))

    hr, y, u = collect(rec_model, loader(), rec_model.tok, maxlen)
    w_rec = rec_model.mt5.lm_head.weight.detach().float().cpu().clone()
    del rec_model
    torch.cuda.empty_cache()
    print(f"recipient states {tuple(hr.shape)}")

    if a.donor:
        don_model, don_ck, don_path = build(a.donor)
        if don_ck["args"]["fold"] != fold:
            raise SystemExit(f"donor is fold {don_ck['args']['fold']}, recipient "
                             f"fold {fold} — the dev splits would differ")
        hd, y2, _ = collect(don_model, loader(), don_model.tok, maxlen)
        w_don = don_model.mt5.lm_head.weight.detach().float().cpu().clone()
        del don_model
        torch.cuda.empty_cache()
        if not torch.equal(y, y2):
            raise SystemExit("label alignment differs between the two passes")
        print(f"donor states {tuple(hd.shape)}")
        donor_name = a.donor
    elif a.donor_head:
        hd = None
        w_don = head_of_any(Path(a.donor_head))
        donor_name = Path(a.donor_head).stem.replace(".run", "")
        bad = [m for m in a.methods.split(",")
               if m.strip() in ("procrustes", "ridge")]
        if bad:
            raise SystemExit(f"{bad} need donor hidden states; pass --donor, or "
                             "restrict --methods to learned")
        print(f"donor head only: {donor_name} (no donor model, learned map only)")
    else:
        raise SystemExit("pass --donor or --donor-head")

    n = hr.shape[0]
    g = torch.Generator().manual_seed(0)
    utts = torch.unique(u)
    uperm = utts[torch.randperm(len(utts), generator=g)]
    ho_utts = set(uperm[: max(1, int(len(utts) * a.holdout))].tolist())
    is_ho = torch.tensor([int(x) in ho_utts for x in u])
    ho = torch.nonzero(is_ho, as_tuple=True)[0]
    fit = torch.nonzero(~is_ho, as_tuple=True)[0]
    n_ho = len(ho)
    print(f"{len(fit)} fit / {n_ho} held-out positions "
          f"({len(utts) - len(ho_utts)} / {len(ho_utts)} utterances)")

    rep = {"recipient": a.recipient, "donor": donor_name, "fold": fold,
           "donor_is_run": bool(a.donor),
           "n_positions": n, "n_fit": len(fit), "n_holdout": n_ho,
           "state_cos_mean": None if hd is None else float(
               torch.nn.functional.cosine_similarity(hr, hd, dim=1).mean()),
           "state_rel_l2": None if hd is None else float(
               (hr - hd).norm() / hd.norm()),
           "head_cos": float(torch.nn.functional.cosine_similarity(
               w_rec.double().flatten(), w_don.double().flatten(), dim=0)),
           "ce_own_head": teacher_ce(hr, y, w_rec),
           "ce_donor_head_raw": teacher_ce(hr, y, w_don),
           "ce_donor_on_donor": None if hd is None else teacher_ce(hd, y, w_don),
           "methods": {}}
    print(f"\nstate cos {rep['state_cos_mean']}  rel L2 {rep['state_rel_l2']}"
          f"   head cos {rep['head_cos']:.4f}")
    print(f"dev CE  own head {rep['ce_own_head']:.4f}   donor head raw "
          f"{rep['ce_donor_head_raw']:.4f}   donor on its own states "
          f"{rep['ce_donor_on_donor']}")

    DONORS.mkdir(parents=True, exist_ok=True)
    for method in [m.strip() for m in a.methods.split(",") if m.strip()]:
        print(f"\n--- {method}")
        if method == "procrustes":
            m = fit_procrustes(hr[fit], hd[fit])
        elif method == "ridge":
            m = fit_ridge(hr[fit], hd[fit], a.ridge_lam)
        elif method == "learned":
            m, log = fit_learned(hr[fit], y[fit], w_don, a.steps, a.lr,
                                 a.fit_batch)
        else:
            raise SystemExit(f"unknown method {method!r}")

        # Fit quality, held out. For the two imitation maps this is how well the
        # recipient's states can be made to look like the donor's; for `learned`
        # it is beside the point, and the CE below is the number that matters.
        resid = r2 = None
        if hd is not None:
            resid = float((hr[ho] @ m - hd[ho]).norm() / hd[ho].norm())
            r2 = 1.0 - resid ** 2
        ce = teacher_ce(hr[ho], y[ho], w_don, m)
        ce_raw = teacher_ce(hr[ho], y[ho], w_don)
        ce_own = teacher_ce(hr[ho], y[ho], w_rec)
        name = f"align_{method}_{a.recipient}__{donor_name}"
        w_new = (w_don @ m).contiguous()
        torch.save({"arch": "unisign", "state": {RUN_KEY: w_new},
                    "cfg": {"synthetic_head": name}}, DONORS / f"{name}.run.pth")
        rec = {"holdout_rel_resid": resid, "holdout_r2": r2,
               "holdout_ce_donor_head_aligned": ce,
               "holdout_ce_donor_head_raw": ce_raw,
               "holdout_ce_own_head": ce_own,
               "m_fro": float(m.norm()),
               "m_dist_from_identity": float(
                   (m - torch.eye(m.shape[0])).norm() / m.shape[0] ** 0.5),
               "donor_run": f"{name}.run.pth"}
        if method == "learned":
            rec["log"] = log
        rep["methods"][method] = rec
        print(f"  held-out: rel resid {resid} (R2 {r2})   "
              f"CE aligned {ce:.4f} vs raw {ce_raw:.4f} vs own head {ce_own:.4f}")
        print(f"  -> {DONORS / (name + '.run.pth')}")

    prev = json.loads(a.out.read_text()) if a.out.exists() else {}
    prev[f"{a.recipient}__{donor_name}"] = rep
    a.out.write_text(json.dumps(prev, ensure_ascii=False, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
