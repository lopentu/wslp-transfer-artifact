#!/usr/bin/env python3
"""Fetch Uni-Sign's source into third_party/ so we can import their model.

**Licensing.** The Uni-Sign repository publishes no licence file, and the GitHub
API reports no licence, which means all rights are reserved by default. So their
code is *fetched* at setup time into an untracked directory and imported — never
copied into this repository and never redistributed with it.

The checkpoints are fetched separately by `08_check_external.py`.

    python3 scripts/09_fetch_unisign.py
    python3 scripts/09_fetch_unisign.py --ref <commit-sha>   # pin a revision
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/ZechengLi19/Uni-Sign/{ref}/{path}"

# Everything the model definition transitively needs. datasets.py is included for
# `load_part_kp` / `crop_scale`: those define the exact normalisation the released
# weights were trained under, and reimplementing them by eye would risk a silent
# numerical mismatch that looks like a bad transfer result.
FILES = [
    "models.py",
    "datasets.py",
    "config.py",
    "utils.py",
    "deformable_attention_2d.py",
    "stgcn_layers/__init__.py",
    "stgcn_layers/gcn_utils.py",
    "stgcn_layers/stgcn_block.py",
]

NOTE = """This directory holds third-party source fetched by
scripts/09_fetch_unisign.py from https://github.com/ZechengLi19/Uni-Sign

That repository carries NO licence, so these files are all-rights-reserved by
their authors. They are fetched locally for research use and are deliberately
NOT tracked by git and NOT redistributed. Do not commit them.

Fetched ref: {ref}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    # 8/20: pinned. `main` is a moving target, so "the code we ran" was not a
    # defined artifact and the paper could not name one. This SHA is not a guess:
    # every file in third_party/unisign/ was verified byte-identical to it
    # (config.py after undoing our own mt5_path rewrite), and it is the newest
    # commit at or before the 8/11 fetch. Pass --ref to override.
    ap.add_argument("--ref", default="eed438bcb49e30405cd6ccdfcccca330c134e830",
                    help="branch or commit sha (default: the pinned, verified commit)")
    ap.add_argument("--out", type=Path, default=Path("third_party/unisign"))
    ap.add_argument("--mt5", default="google/mt5-base",
                    help="what to rewrite config.mt5_path to; their default is a "
                         "local ./pretrained_weight/mt5-base that we do not have")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    for rel in FILES:
        dest = a.out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = RAW.format(ref=a.ref, path=rel)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                dest.write_bytes(r.read())
        except Exception as e:
            print(f"FAILED {rel}: {type(e).__name__}: {e}")
            sys.exit(1)
        print(f"  {rel:34s} {dest.stat().st_size:7d} B")

    # Their config.py points mt5_path at a local checkout we do not have; make it
    # a hub id so `MT5ForConditionalGeneration.from_pretrained` just works.
    cfg = a.out / "config.py"
    text = cfg.read_text()
    if "./pretrained_weight/mt5-base" in text:
        cfg.write_text(text.replace('mt5_path = "./pretrained_weight/mt5-base"',
                                    f'mt5_path = "{a.mt5}"'))
        print(f"  config.py: mt5_path -> {a.mt5}")

    (a.out / "README.NOTICE").write_text(NOTE.format(ref=a.ref))
    gi = Path("third_party/.gitignore")
    gi.parent.mkdir(parents=True, exist_ok=True)
    gi.write_text("*\n!.gitignore\n")
    print(f"\nfetched to {a.out} (untracked; see README.NOTICE)")
    print("next: python3 -c \"import sys; sys.path.insert(0,'src'); "
          "from tsl.unisign import selftest; selftest()\"")


if __name__ == "__main__":
    main()
