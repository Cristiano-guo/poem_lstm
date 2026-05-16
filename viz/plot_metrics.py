# -*- coding: utf-8 -*-
"""根据 summary.csv 绘制各方法指标柱状图 + 雷达图。"""
from __future__ import annotations

import argparse
import csv
import math
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


def load_summary(p: Path):
    rows = []
    with p.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            for k, v in list(row.items()):
                if k == "model":
                    continue
                try:
                    row[k] = float(v) if v else 0.0
                except ValueError:
                    pass
            rows.append(row)
    return rows


def plot_bar(rows, out: Path) -> None:
    metrics = ["format_acc", "rhyme_acc", "distinct_2", "bleu_2", "rouge_l"]
    metrics = [m for m in metrics if any(m in r for r in rows)]
    models = [r["model"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.2), 5))
    x = np.arange(len(models))
    width = 0.15
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, [r.get(m, 0) for r in rows], width, label=m)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("各方法指标对比")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[done] -> {out}")


def plot_radar(rows, out: Path) -> None:
    # 选 5 个 0-1 区间指标
    metrics = ["format_acc", "rhyme_acc", "distinct_2", "bleu_2", "rouge_l"]
    metrics = [m for m in metrics if any(m in r for r in rows)]
    if not metrics:
        return
    angles = np.linspace(0, 2 * math.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for r in rows:
        vals = [r.get(m, 0) for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, marker="o", label=r["model"])
        ax.fill(angles, vals, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title("自动指标雷达图")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.05))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[done] -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="eval/results/summary.csv")
    parser.add_argument("--out_bar", default="viz/metric_bar.png")
    parser.add_argument("--out_radar", default="viz/radar_chart.png")
    args = parser.parse_args()

    rows = load_summary(Path(args.summary))
    if not rows:
        print("[warn] summary 为空")
        return
    Path(args.out_bar).parent.mkdir(parents=True, exist_ok=True)
    plot_bar(rows, Path(args.out_bar))
    plot_radar(rows, Path(args.out_radar))


if __name__ == "__main__":
    main()
