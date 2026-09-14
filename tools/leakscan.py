"""Does a tree contain Taiwan Sign Language Corpus material?

The paper undertakes to redistribute no corpus material or signer metadata.
That undertaking is checkable rather than merely asserted: build the set of
corpus utterances from an unredacted item file, then look for any of them
inside every file of the tree under audit. A hit is a file that cannot ship.

This is the check run against the unredacted corpus as the last step of
packaging, and it is shipped so that the claim can be audited rather than
taken on trust. Auditing it needs a corpus licence, because the inventory it
compares against is exactly what the release withholds:

    python tools/leakscan.py --records /path/to/unredacted/records.jsonl \\
        'data/*.json' 'runs/*/*.json' 'src/tsl/*.py'

Pointed at the released item file it will refuse to run, rather than compare
redacted ids against redacted ids and report a reassuring nothing.
"""
import argparse
import glob
import json
import os
import re
import sys

HAN = re.compile("[\\u4e00-\\u9fff]")
MIN_LEN = 4          # shorter than this and a "match" is a common collocation


def inventory(path):
    """Every utterance the corpus contributed, from an unredacted item file.

    Redaction replaces each sentence with an opaque id, so a redacted file
    still has `candidates` and would still yield a set of strings. `text` is
    the field redaction drops outright, and its absence is what distinguishes
    the two cases.
    """
    sents, redacted, total = set(), 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            total += 1
            if "text" not in d:
                redacted += 1
                continue
            for s in ([d.get("text")]
                      + list(d.get("candidates") or [])
                      + [c.get("text") for c in (d.get("ctx") or [])]):
                if s and len(s) >= MIN_LEN:
                    sents.add(s)
    return sents, redacted, total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("patterns", nargs="+", help="globs of files to audit")
    ap.add_argument("--records", default="data/lex/records.jsonl",
                    help="an UNREDACTED records.jsonl to build the inventory from")
    a = ap.parse_args()

    if not os.path.exists(a.records):
        sys.exit(f"no item file at {a.records}; pass --records")
    sents, redacted, total = inventory(a.records)
    if not sents:
        sys.exit(
            f"{a.records} is the redacted item file ({redacted} of {total} records\n"
            "carry no utterance), so there is no inventory to compare against and\n"
            "this scan would be meaningless. Point --records at an unredacted copy;\n"
            "that it is missing here is the release working as intended.")
    print(f"inventory: {len(sents)} corpus utterances of {MIN_LEN}+ characters")

    by_prefix = {}
    for s in sents:
        by_prefix.setdefault(s[:MIN_LEN], []).append(s)

    leaks = 0
    for pattern in a.patterns:
        for p in sorted(glob.glob(pattern, recursive=True)):
            if not os.path.isfile(p):
                continue
            try:
                blob = open(p, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                continue
            hits = set()
            for i in range(len(blob) - MIN_LEN + 1):
                for s in by_prefix.get(blob[i:i + MIN_LEN], ()):
                    if blob.startswith(s, i):
                        hits.add(s)
            if hits:
                leaks += 1
                print(f"!! LEAK  {p}")
                for s in sorted(hits, key=len, reverse=True)[:3]:
                    print(f"         {s[:44]}")
            else:
                print(f"   clean  han={len(HAN.findall(blob)):<6d} {p}")
    print(f"\nfiles containing a corpus utterance: {leaks}")
    return 1 if leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
