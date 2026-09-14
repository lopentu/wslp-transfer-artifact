# SignCLIP probe environment

Stage 2 of the JSL probe (`experiment-suggestion-first.md` §3). Everything here
lives in **`.venv_signclip`, never the training venv** — `pose_format` pins
`mediapipe<0.10.30` and `fairseq` pins `hydra-core<1.1`, either of which would
downgrade a dependency out from under a running sweep.

## Why the probe is clean

SignCLIP was pretrained on SpreadTheSign, whose language inventory (verified
2026-08-12 from the site's own language selector) includes **Japanese, Chinese
(China) and American ("English (United States)") sign languages, and does not
include Taiwanese**. TSL is therefore genuinely unseen data for it. Because the
model also accepts language-tagged text prompts (`<en> <ase> house`), the
cross-language comparison needs no non-TSL video at all.

`vfeat_dim: 609` in the model config = 203 keypoints × 3, which is exactly what
`preprocess_pose` reduces MediaPipe Holistic to. Our extracted `.pose` files
carry all four components it asks for, so they feed in unmodified.

## Getting it to install (four separate blockers)

Reproducing this from scratch is not obvious; each step below failed loudly
before the next was found.

1. **`pip install -e` pulled a CUDA torch.** Build isolation resolves torch from
   PyPI, ignoring a `--index-url` given to an earlier command, and dragged in
   ~1.5 GB of `nvidia_*` wheels. This filled the root filesystem. Install CPU
   torch first, then build with **`--no-build-isolation`** so it reuses it.
2. **`omegaconf 2.0.5` has invalid metadata** (`PyYAML (>=5.1.*)`), which pip
   ≥24.1 rejects outright. Pin **`pip<24.1`** inside this venv.
3. **Missing Cython-generated sources** — fairseq ships `.pyx` and the build
   needs `cython` present to emit `data_utils_fast.cpp`. `pip install cython`.
4. **hydra 1.0.7 vs Python 3.12 dataclasses.** Python ≥3.11 rejects a mutable
   default where `default_factory` is required, so importing `mmpt` dies on
   `hydra.conf`. fairseq pins `hydra-core<1.1` so upgrading is not available.
   Fix: patch the nine `x: T = T()` defaults in
   `.venv_signclip/lib/python3.12/site-packages/hydra/conf/__init__.py` to
   `field(default_factory=T)`.

   **This patch is lost if `.venv_signclip` is rebuilt.** Re-apply with the
   regex in this file's git history, or by hand — the nine lines are all of the
   form `name: Type = Type()` in that one file.

## Running it without disturbing the shared GPU

`demo_sign.get_model()` calls `.cuda()` whenever `torch.cuda.is_available()`.
A colleague holds the card 21:00–09:00, so:

- torch here is **`2.6.0+cpu`**, so `cuda.is_available()` is `False` and the
  model *cannot* reach the GPU. This is the structural guarantee.
- Still pass `CUDA_VISIBLE_DEVICES=""` as a second layer.
- Cap CPU threads. Their dataloaders need cores too; this box ran at load ~63/32
  during our own sweep, and an unbounded thread pool here would slow their job.

## Checkpoint layout

`MMPTModel.from_pretrained` resolves weights to `config.eval.save_path` +
`checkpoint_best.pt`, i.e. relative to `examples/MMPT/`:

    third_party/signclip_fairseq/examples/MMPT/runs/retri_v1_1/baseline_temporal/checkpoint_best.pt

Weights come from the authors' Google Drive folder
`10q7FxPlicrfwZn7_FgtNqKFDiAJi6CTc` (linked from `examples/MMPT/README.md`).

## Pose store

`scripts/13_signclip_poses.py` builds `data/signclip/pose/*.pose` from the twtsl
citation clips. 3,500 clips, 0 errors, 3.2 GB, ~59 min on 8 niced workers.

Quality, measured over the window where the sign actually occurs (whole-clip
means are misleading — dictionary clips open and close on a rest pose, which
drags the average to ~40%): **67% of clips have ≥90% hand-landmark coverage, 88%
have ≥50%, and only 3.8% have no usable hands at all.**
