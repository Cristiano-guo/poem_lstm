# -*- coding: utf-8 -*-
"""GPT-2 微调数据集：把诗包装成 "<体裁>正文<eos>" 的简短样本。"""
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset


FORMAT_TAGS = {
    "tang_jueju5.train.txt": "[五绝]",
    "tang_jueju7.train.txt": "[七绝]",
    "tang_lvshi5.train.txt": "[五律]",
    "tang_lvshi7.train.txt": "[七律]",
    "ci_song.train.txt": "[词]",
}


class PoemSFTDataset(Dataset):
    def __init__(self, files: List[Path], tokenizer, max_length: int = 96) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[str] = []
        for f in files:
            if not f.exists():
                print(f"[warn] {f} 不存在，跳过")
                continue
            tag = FORMAT_TAGS.get(f.name, "")
            for ln in f.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                # 词的格式是 "<词牌>\t<正文>"
                if "\t" in ln:
                    rhythmic, body = ln.split("\t", 1)
                    text = f"{tag}《{rhythmic}》{body}"
                else:
                    text = f"{tag}{ln}"
                self.samples.append(text)
        print(f"[info] PoemSFTDataset: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        text = self.samples[idx]
        ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=self.max_length - 1,
            truncation=True,
        )
        ids = ids + [self.tokenizer.eos_token_id or self.tokenizer.sep_token_id]
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch, pad_id: int):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), maxlen), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = b["input_ids"]
        labels[i, :n] = b["labels"]
        attn[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}
