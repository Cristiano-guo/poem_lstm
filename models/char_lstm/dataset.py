# -*- coding: utf-8 -*-
"""Char-LSTM 训练数据集。"""
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

from .vocab import BOS, EOS, Vocab


def load_lines(paths: List[Path]) -> List[str]:
    """读取多个 .txt 文件，每行一首诗，合并返回。"""
    lines: List[str] = []
    for p in paths:
        if not p.exists():
            print(f"[warn] {p} 不存在，跳过")
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                lines.append(ln)
    return lines


class PoemDataset(Dataset):
    """每首诗包装成 [BOS] ... [EOS]，按 seq_len 切窗口。

    构造时把所有诗序列化为一个长 token 流（每首诗以 BOS/EOS 包围），
    再切成定长样本，每个样本同时给出 input 与 target（错位 1 位）。
    """

    def __init__(self, lines: List[str], vocab: Vocab, seq_len: int = 48) -> None:
        self.seq_len = seq_len
        self.vocab = vocab

        # 整条流：每首诗以 BOS 开头、EOS 结尾
        stream: List[int] = []
        for ln in lines:
            stream.append(vocab.bos_id)
            stream.extend(vocab.stoi.get(ch, vocab.unk_id) for ch in ln)
            stream.append(vocab.eos_id)
        self.data = torch.tensor(stream, dtype=torch.long)

    def __len__(self) -> int:
        # 留 1 位给 target
        return max(0, (self.data.numel() - 1) // self.seq_len)

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.data[start:end]
        y = self.data[start + 1 : end + 1]
        return x, y
