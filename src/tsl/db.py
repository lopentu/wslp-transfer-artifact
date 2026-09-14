"""Access layer for the 文化部 臺灣手語語料庫 capture (`tslcorpus.db`).

Encodes the two traps documented in the corpus README so no call site has to
remember them:

* ``hand_tier = 2`` is a byte-exact duplicate of tier 1 -> always filter tier 1.
* the eleven annotation layers are *not* index-aligned with ``tokens`` -> join on
  ``(uuid, seq, t1, t2)``, never on ``idx``.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

CORPUS_DIR = Path("/mnt/md0/corpus/sign/tslcorpus")
DB_PATH = CORPUS_DIR / "tslcorpus.db"
FILMS_DIR = CORPUS_DIR / "films"

POS_LAYER = "詞類"

# Pointing signs. The split is the whole point of the diagnostic: 1st/2nd person
# loci are fixed by the participants physically present, so a single clip is
# enough. 3rd person loci are assigned arbitrarily in the signing space earlier
# in the discourse, so a single clip is *not* enough.
DEICTIC = {"我", "你", "妳", "我們", "你們", "我們兩個", "我們兩人"}
ANAPHORIC = {"他", "她", "他們", "她們", "牠", "他們兩個", "她們兩個", "他們倆個"}
# 自己/大家 are reflexive/collective — resolvable either way, kept out of both.
PRONOUNS = DEICTIC | ANAPHORIC

# 呼應動詞 / 空間動詞 lifted from tsldict (the tslcorpus 動詞類型 layer is 0% filled).
# Path is resolved lazily by `agreeing_verbs()`.
TSLDICT_DB = Path("/mnt/md0/corpus/sign/tsldict/tsldict.db")

# The one film that is truncated at source: 46.1 s of video against 51.9 s of
# annotation. Anything cutting by t1/t2 must clamp or drop.
TRUNCATED = {"G4C26": 46_100}

SPEAKER_SIDES = ("L", "R")

# 17 dialogue sentences carry no usable speaker: three are empty and fourteen hold
# a transcription of the sentence pasted into the column (a column-shift during
# annotation, not an "unknown"). Left as None they are silently unextractable,
# because a dialogue film only has crops for L and R. scripts/10_recover_speakers.py
# infers the side from which half of the frame moves and writes this sidecar; it is
# merged here rather than edited into the corpus, and marked so it stays visible.
SPEAKER_SIDECAR = Path("data/speaker_inferred.json")


@dataclass
class Sentence:
    uuid: str
    seq: int
    t1: int
    t2: int
    text: str
    speaker: str | None  # 'L' | 'R' | None (monologue)
    para_type: str  # '1' = 篇章 monologue, '2' = 對話 dialogue
    theme: str
    tokens: list[tuple[int, str, str]] = field(default_factory=list)  # (idx, word, pos)
    speaker_src: str = "annotated"  # 'annotated' | 'motion' — never conflate them

    @property
    def key(self) -> str:
        return f"{self.uuid}:{self.seq:03d}"

    @property
    def dur_ms(self) -> int:
        return self.t2 - self.t1


class Corpus:
    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = Path(db_path)
        self.con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        self.con.row_factory = sqlite3.Row

    # ---------------------------------------------------------------- metadata

    @cached_property
    def paragraphs(self) -> dict[str, dict]:
        out = {}
        for r in self.con.execute(
            "SELECT uuid, type, name, theme_name, attr FROM paragraphs"
        ):
            try:
                attr = json.loads(r["attr"] or "[]")
            except json.JSONDecodeError:
                attr = []
            out[r["uuid"]] = {
                "uuid": r["uuid"],
                "type": r["type"],
                "name": r["name"],
                "theme": r["theme_name"] or "",
                "signer": next((a for a in attr if a.startswith("演繹者")), ""),
                "sex": next((a for a in attr if a in ("男性", "女性")), ""),
                "age": next((a for a in attr if a.endswith("歲") or "歲" in a), ""),
            }
        return out

    # --------------------------------------------------------------- sentences

    @cached_property
    def sentences(self) -> list[Sentence]:
        toks = self._tokens_by_sentence()
        rows = self.con.execute(
            """SELECT s.uuid, s.seq, s.t1, s.t2, s.text, s.speaker, p.type, p.theme_name
               FROM sentences s JOIN paragraphs p USING (uuid)
               WHERE s.t1 IS NOT NULL AND s.t2 > s.t1
               ORDER BY s.uuid, s.seq"""
        )
        inferred = self._inferred_speakers
        out = []
        for r in rows:
            spk = r["speaker"] if r["speaker"] in SPEAKER_SIDES else None
            src = "annotated"
            if spk is None and r["type"] == "2":
                guess = inferred.get(f"{r['uuid']}:{r['seq']:03d}")
                if guess:
                    spk, src = guess["speaker"], "motion"
            out.append(
                Sentence(
                    uuid=r["uuid"],
                    seq=r["seq"],
                    t1=r["t1"],
                    t2=r["t2"],
                    text=(r["text"] or "").strip(),
                    speaker=spk,
                    para_type=r["type"],
                    theme=r["theme_name"] or "",
                    tokens=toks.get((r["uuid"], r["seq"]), []),
                    speaker_src=src,
                )
            )
        return out

    @cached_property
    def _inferred_speakers(self) -> dict[str, dict]:
        """Sidecar written by scripts/10_recover_speakers.py; absent is fine."""
        if not SPEAKER_SIDECAR.exists():
            return {}
        return json.loads(SPEAKER_SIDECAR.read_text())

    def _tokens_by_sentence(self) -> dict[tuple[str, int], list[tuple[int, str, str]]]:
        """Sign tokens with their 詞類 tag, joined on timestamps (not idx)."""
        rows = self.con.execute(
            """SELECT t.uuid, t.seq, t.idx, t.word, a.value AS pos
               FROM tokens t
               LEFT JOIN annotations a
                 ON  a.uuid = t.uuid AND a.seq = t.seq
                 AND a.t1 = t.t1 AND a.t2 = t.t2 AND a.layer = ?
               WHERE t.hand_tier = 1
               ORDER BY t.uuid, t.seq, t.idx""",
            (POS_LAYER,),
        )
        out: dict[tuple[str, int], list[tuple[int, str, str]]] = {}
        for r in rows:
            out.setdefault((r["uuid"], r["seq"]), []).append(
                (r["idx"], (r["word"] or "").strip(), (r["pos"] or "").strip())
            )
        return out

    def by_paragraph(self) -> dict[str, list[Sentence]]:
        out: dict[str, list[Sentence]] = {}
        for s in self.sentences:
            out.setdefault(s.uuid, []).append(s)
        for v in out.values():
            v.sort(key=lambda s: s.seq)
        return out

    # ------------------------------------------------------------------ films

    def film(self, uuid: str) -> Path:
        return FILMS_DIR / f"{uuid}.mp4"

    def clamp(self, uuid: str, t1: int, t2: int) -> tuple[int, int] | None:
        """Clamp a span to the film, returning None if nothing survives."""
        limit = TRUNCATED.get(uuid)
        if limit is None:
            return (t1, t2)
        if t1 >= limit:
            return None
        return (t1, min(t2, limit))


def agreeing_verbs(db_path: Path | str = TSLDICT_DB) -> set[str]:
    """呼應動詞 / 空間動詞 glosses from tsldict — verbs whose *direction* carries
    argument structure, so their reading depends on where the loci were set."""
    p = Path(db_path)
    if not p.exists():
        return set()
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT word FROM words WHERE verb_type IN ('呼應動詞','空間動詞')"
    )
    out = set()
    for (w,) in rows:
        w = (w or "").strip()
        if not w:
            continue
        out.add(w)
        # dictionary head words carry disambiguators: 幫、幫忙、幫助(北), 買(a)
        head = w.split("(")[0]
        for part in head.replace("、", "、").split("、"):
            part = part.strip()
            if len(part) >= 1:
                out.add(part)
    return out
