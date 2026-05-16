# -*- coding: utf-8 -*-
"""Char-level 词表构建与序列化。"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List

PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]


class Vocab:
    def __init__(self, itos: List[str]):
        self.itos: List[str] = list(itos)
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.stoi.get(ch, self.unk_id) for ch in text)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], strip_special: bool = True) -> str:
        out = []
        special = {self.pad_id, self.bos_id, self.eos_id}
        for i in ids:
            if strip_special and i in special:
                continue
            out.append(self.itos[i] if 0 <= i < len(self.itos) else "")
        return "".join(out)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.itos, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            itos = json.load(f)
        return cls(itos)

    @classmethod
    def build(cls, texts: Iterable[str], min_count: int = 1) -> "Vocab":
        cnt: Counter[str] = Counter()
        for t in texts:
            cnt.update(t)
        itos = list(SPECIAL_TOKENS)
        for ch, c in cnt.most_common():
            if c < min_count:
                continue
            if ch in SPECIAL_TOKENS:
                continue
            itos.append(ch)
        return cls(itos)
