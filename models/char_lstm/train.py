# -*- coding: utf-8 -*-
"""Char-LSTM 训练脚本。

用法：
    python models/char_lstm/train.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# 允许以脚本方式运行
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.char_lstm.dataset import PoemDataset, load_lines
from models.char_lstm.model import CharLSTM
from models.char_lstm.vocab import Vocab


def evaluate(model: CharLSTM, loader: DataLoader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    n_tokens = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            ntok = (y != model.pad_id).sum().item()
            total_loss += loss.item() * ntok
            n_tokens += ntok
    avg = total_loss / max(1, n_tokens)
    return avg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out_dir", default="ckpts/char_lstm")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    lcfg = cfg["lstm"]

    processed_root = Path(cfg["paths"]["processed_root"]).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_files = [
        processed_root / f.replace(".txt", ".train.txt") for f in lcfg["train_files"]
    ]
    val_files = [
        processed_root / f.replace(".txt", ".val.txt") for f in lcfg["train_files"]
    ]

    print(f"[info] train_files = {[str(p) for p in train_files]}")
    print(f"[info] val_files   = {[str(p) for p in val_files]}")

    train_lines = load_lines(train_files)
    val_lines = load_lines(val_files)
    print(f"[info] train_lines = {len(train_lines)}; val_lines = {len(val_lines)}")

    if not train_lines:
        raise RuntimeError("训练数据为空，请先运行 data/preprocess.py")

    # 词表：从训练集字符 + 标点
    vocab = Vocab.build(train_lines, min_count=lcfg.get("vocab_min_count", 1))
    print(f"[info] vocab_size = {len(vocab)}")
    vocab.save(out_dir / "vocab.json")

    train_ds = PoemDataset(train_lines, vocab, seq_len=lcfg["seq_len"])
    val_ds = PoemDataset(val_lines, vocab, seq_len=lcfg["seq_len"]) if val_lines else None

    train_loader = DataLoader(
        train_ds,
        batch_size=lcfg["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=lcfg["batch_size"], shuffle=False, num_workers=0)
        if val_ds is not None
        else None
    )
    print(f"[info] train_batches = {len(train_loader)}")

    device = torch.device(lcfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    model = CharLSTM(
        vocab_size=len(vocab),
        embedding_dim=lcfg["embedding_dim"],
        hidden_dim=lcfg["hidden_dim"],
        num_layers=lcfg["num_layers"],
        dropout=lcfg["dropout"],
        pad_id=vocab.pad_id,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=lcfg["lr"])

    history = {"train_loss": [], "val_loss": [], "val_ppl": []}
    best_val = float("inf")

    for epoch in range(1, lcfg["epochs"] + 1):
        model.train()
        ep_loss = 0.0
        ep_tok = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), lcfg["grad_clip"])
            optimizer.step()
            ntok = (y != vocab.pad_id).sum().item()
            ep_loss += loss.item() * ntok
            ep_tok += ntok
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = ep_loss / max(1, ep_tok)
        history["train_loss"].append(train_loss)

        log = f"[epoch {epoch}] train_loss={train_loss:.4f} ppl={math.exp(train_loss):.2f} time={time.time()-t0:.1f}s"
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_ppl"].append(math.exp(val_loss))
            log += f" | val_loss={val_loss:.4f} val_ppl={math.exp(val_loss):.2f}"
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "vocab_size": len(vocab),
                        "config": lcfg,
                    },
                    out_dir / "best.pt",
                )
                log += "  [saved best]"
        print(log)

    # 保存最终模型 + history
    torch.save(
        {"model": model.state_dict(), "vocab_size": len(vocab), "config": lcfg},
        out_dir / "last.pt",
    )
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
