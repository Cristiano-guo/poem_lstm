# -*- coding: utf-8 -*-
"""根据填写好的 human_eval.csv 绘制 5 维度人工评价雷达图。"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

DIMS = ["fluency_1to5", "relevance_1to5", "format_1to5", "style_1to5", "poetic_1to5"]
DIM_NAMES = ["流畅性", "相关性", "格式", "风格", "诗意"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="eval/results/human_eval.csv")
    parser.add_argument("--out", default="viz/human_radar.png")
    args = parser.parse_args()

    model2scores = defaultdict(lambda: defaultdict(list))
    with open(args.csv, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            m = row.get("model", "")
            for d in DIMS:
                v = row.get(d, "")
                try:
                    v = float(v)
                except ValueError:
                    continue
                model2scores[m][d].append(v)

    if not model2scores:
        print("[warn] CSV 中没有有效评分")
        return

    angles = np.linspace(0, 2 * math.pi, len(DIMS), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for model, dscores in model2scores.items():
        vals = []
        for d in DIMS:
            arr = dscores.get(d, [])
            vals.append(sum(arr) / len(arr) if arr else 0.0)
        vals += vals[:1]
        ax.plot(angles, vals, marker="o", label=model)
        ax.fill(angles, vals, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DIM_NAMES)
    ax.set_ylim(0, 5)
    ax.set_title("人工评价雷达图 (1-5 分制)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.05))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
