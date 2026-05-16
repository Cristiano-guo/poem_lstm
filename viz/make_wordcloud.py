# -*- coding: utf-8 -*-
"""根据训练语料生成中文高频字 / 词词云。"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


_HAN = re.compile(r"[\u4e00-\u9fff]")

# 常见过滤字（虚词、标点）
_STOPWORDS = set("一二三四五六七八九十百千万的了是不也在与之而其于以为又有又又")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/processed/tang_jueju5.txt",
            "data/processed/tang_jueju7.txt",
        ],
    )
    parser.add_argument("--out", default="viz/wordcloud.png")
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--font", default=None, help="可选字体文件路径")
    args = parser.parse_args()

    text_all = ""
    for p in args.inputs:
        fp = Path(p)
        if not fp.exists():
            continue
        text_all += fp.read_text(encoding="utf-8")

    chars = [c for c in text_all if _HAN.match(c) and c not in _STOPWORDS]
    cnt = Counter(chars)
    top = dict(cnt.most_common(args.top))

    try:
        from wordcloud import WordCloud

        font_path = args.font
        if font_path is None:
            # 尝试常见中文字体
            for cand in [
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simhei.ttf",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            ]:
                if Path(cand).exists():
                    font_path = cand
                    break
        wc = WordCloud(
            font_path=font_path,
            width=1200,
            height=600,
            background_color="white",
            max_words=args.top,
            collocations=False,
        ).generate_from_frequencies(top)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        print(f"[done] -> {args.out}")
    except ImportError:
        # 回退：柱状图
        print("[warn] wordcloud 未安装，用柱状图代替")
        items = list(top.items())[:30]
        items.sort(key=lambda x: x[1], reverse=True)
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(labels, values)
        ax.set_title("Top-30 高频字")
        plt.xticks(rotation=0)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)


if __name__ == "__main__":
    main()
