# -*- coding: utf-8 -*-
"""绘制训练 loss 曲线（兼容 char_lstm 与 gpt2_lora 的 history.json）。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        nargs="+",
        required=True,
        help="history.json 路径列表，可多个；图例使用父目录名",
    )
    parser.add_argument("--out", default="viz/loss_curve.png")
    args = parser.parse_args()

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    for hp in args.history:
        with open(hp, "r", encoding="utf-8") as f:
            h = json.load(f)
        name = Path(hp).parent.name
        tr = h.get("train_loss", [])
        ax[0].plot(range(1, len(tr) + 1), tr, marker="o", label=f"{name} train")
        if "val_loss" in h and h["val_loss"]:
            ax[0].plot(range(1, len(h["val_loss"]) + 1), h["val_loss"], marker="s", label=f"{name} val")
        ppl = [math.exp(x) for x in tr]
        ax[1].plot(range(1, len(ppl) + 1), ppl, marker="o", label=f"{name} train PPL")

    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss (CE)")
    ax[0].set_title("训练 / 验证 损失")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Perplexity")
    ax[1].set_title("训练困惑度")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
