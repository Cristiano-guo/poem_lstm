# -*- coding: utf-8 -*-
"""
chinese-poetry 数据预处理脚本

功能：
1) 遍历 全唐诗/poet.tang.*.json、宋词/ci.song.*.json
2) 繁体 -> 简体（OpenCC）
3) 按格式分桶：五言绝句 / 七言绝句 / 五言律诗 / 七言律诗 / 宋词
4) 清洗：去除含特殊字符、长度异常、重复的诗
5) 划分训练 / 验证集
6) 顺手生成统计信息（数量、字表大小）

输出：data/processed/{tang_jueju5,tang_jueju7,tang_lvshi5,tang_lvshi7,ci_song}.txt
每行一首诗，句子之间用空格分隔。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Tuple

import yaml
from tqdm import tqdm

try:
    from opencc import OpenCC

    _t2s = OpenCC("t2s")

    def to_simplified(text: str) -> str:
        return _t2s.convert(text)
except Exception:
    print("[warn] opencc 未安装，繁简转换将跳过", file=sys.stderr)

    def to_simplified(text: str) -> str:
        return text


# 允许出现的中文标点
_PUNCT = "，。？！；：、"
# 句末分隔（句号 / 问号 / 感叹号）
_SENT_END = "。？！"
# 中文字符判定
_HAN_RE = re.compile(r"[\u4e00-\u9fff]")
# 整首诗中允许出现的字符集（中文 + 标点）
_VALID_CHARS = re.compile(r"^[\u4e00-\u9fff，。？！；：、]+$")


def _load_json(p: Path) -> list:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_tang(raw_root: Path) -> Iterator[dict]:
    tang_dir = raw_root / "全唐诗"
    if not tang_dir.is_dir():
        print(f"[warn] {tang_dir} 不存在", file=sys.stderr)
        return
    files = sorted(tang_dir.glob("poet.tang.*.json"))
    for f in tqdm(files, desc="全唐诗"):
        try:
            yield from _load_json(f)
        except Exception as e:
            print(f"[warn] {f}: {e}", file=sys.stderr)


def _iter_song_ci(raw_root: Path) -> Iterator[dict]:
    ci_dir = raw_root / "宋词"
    if not ci_dir.is_dir():
        print(f"[warn] {ci_dir} 不存在", file=sys.stderr)
        return
    files = sorted(ci_dir.glob("ci.song.*.json"))
    for f in tqdm(files, desc="宋词"):
        try:
            yield from _load_json(f)
        except Exception as e:
            print(f"[warn] {f}: {e}", file=sys.stderr)


def _split_sentences(paragraphs: List[str]) -> List[str]:
    """把 paragraphs 切成单句（按 。？！），保留末尾标点；丢弃过短残句。"""
    sents = []
    for para in paragraphs:
        # 去掉空白
        para = para.strip()
        if not para:
            continue
        # 按句末符号切分但保留分隔符
        buf = ""
        for ch in para:
            buf += ch
            if ch in _SENT_END:
                if buf.strip():
                    sents.append(buf.strip())
                buf = ""
        if buf.strip():
            sents.append(buf.strip())
    return sents


def _classify_tang(paragraphs: List[str]) -> str | None:
    """根据"句对"结构判定五绝 / 七绝 / 五律 / 七律。

    诗一般每"句"以"，"或"。"结尾。例如"床前明月光，疑是地上霜。"。
    我们以"，"或"。"切分子句，统计：
      - 子句数 = 4 -> 绝句
      - 子句数 = 8 -> 律诗
      - 每个子句字数（去标点后）应一致；常见 5 或 7
    """
    text = "".join(paragraphs)
    if not _VALID_CHARS.match(text):
        return None
    # 按所有标点切，保留标点
    sub_sents = re.split(r"[，。？！；]", text)
    sub_sents = [s for s in sub_sents if s.strip()]
    n = len(sub_sents)
    if n not in (4, 8):
        return None
    lens = [len(s) for s in sub_sents]
    if not lens:
        return None
    if len(set(lens)) != 1:
        return None
    L = lens[0]
    if L == 5 and n == 4:
        return "jueju5"
    if L == 7 and n == 4:
        return "jueju7"
    if L == 5 and n == 8:
        return "lvshi5"
    if L == 7 and n == 8:
        return "lvshi7"
    return None


def _normalize_poem(paragraphs: List[str]) -> str:
    """把 paragraphs 拼成"单行"形式：句间用空格。"""
    text = "".join(paragraphs)
    text = re.sub(r"\s+", "", text)
    # 把所有逗号 / 句号统一为中文逗号 / 句号（保留区分）
    return text


def _normalize_ci(paragraphs: List[str]) -> str:
    text = "".join(paragraphs)
    text = re.sub(r"\s+", "", text)
    return text


def process(raw_root: Path, out_root: Path, cfg: dict) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    # 准备输出 buckets
    buckets: dict[str, list[str]] = {
        "jueju5": [],
        "jueju7": [],
        "lvshi5": [],
        "lvshi7": [],
        "ci": [],
    }
    seen = set()
    stats = Counter()

    min_chars = cfg["preprocess"]["min_chars"]
    max_chars = cfg["preprocess"]["max_chars"]
    to_simp = cfg["preprocess"]["to_simplified"]

    # --------- 唐诗 ---------
    for poem in _iter_tang(raw_root):
        paragraphs = poem.get("paragraphs", [])
        if not paragraphs:
            continue
        text = "".join(paragraphs)
        if to_simp:
            text = to_simplified(text)
            paragraphs = [to_simplified(p) for p in paragraphs]
        if len(text) < min_chars or len(text) > max_chars:
            continue
        fmt = _classify_tang(paragraphs)
        if fmt is None:
            stats["tang_skip"] += 1
            continue
        norm = _normalize_poem(paragraphs)
        if norm in seen:
            stats["dup"] += 1
            continue
        seen.add(norm)
        buckets[fmt].append(norm)
        stats[f"tang_{fmt}"] += 1

    # --------- 宋词 ---------
    for ci in _iter_song_ci(raw_root):
        paragraphs = ci.get("paragraphs", [])
        rhythmic = ci.get("rhythmic", "")
        if not paragraphs:
            continue
        text = "".join(paragraphs)
        if to_simp:
            text = to_simplified(text)
            rhythmic = to_simplified(rhythmic)
        if len(text) < min_chars or len(text) > max_chars:
            continue
        if not _VALID_CHARS.match(text):
            stats["ci_skip"] += 1
            continue
        norm = _normalize_ci([to_simplified(p) for p in paragraphs])
        if norm in seen:
            stats["dup"] += 1
            continue
        seen.add(norm)
        # 词以"<词牌>\t<正文>"格式存储
        buckets["ci"].append(f"{rhythmic}\t{norm}")
        stats["ci_total"] += 1

    # --------- 写文件 ---------
    name_map = {
        "jueju5": "tang_jueju5.txt",
        "jueju7": "tang_jueju7.txt",
        "lvshi5": "tang_lvshi5.txt",
        "lvshi7": "tang_lvshi7.txt",
        "ci": "ci_song.txt",
    }
    for fmt, lines in buckets.items():
        out_path = out_root / name_map[fmt]
        with out_path.open("w", encoding="utf-8") as f:
            for ln in lines:
                f.write(ln + "\n")
        print(f"[ok] {out_path}: {len(lines)} lines")

    # --------- 划分 train/val（按行随机） ---------
    import random

    random.seed(cfg["preprocess"]["random_seed"])
    val_ratio = cfg["preprocess"]["val_ratio"]
    for fmt, fname in name_map.items():
        src = out_root / fname
        lines = src.read_text(encoding="utf-8").splitlines()
        random.shuffle(lines)
        n_val = max(1, int(len(lines) * val_ratio)) if lines else 0
        val = lines[:n_val]
        train = lines[n_val:]
        (out_root / fname.replace(".txt", ".train.txt")).write_text(
            "\n".join(train), encoding="utf-8"
        )
        (out_root / fname.replace(".txt", ".val.txt")).write_text(
            "\n".join(val), encoding="utf-8"
        )

    # --------- 统计 ---------
    stats_path = out_root / "stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(dict(stats), f, ensure_ascii=False, indent=2)
    print(f"[stats] {dict(stats)}")
    print(f"[stats] -> {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_root = Path(cfg["paths"]["raw_root"]).expanduser().resolve()
    out_root = Path(cfg["paths"]["processed_root"]).expanduser().resolve()
    print(f"[info] raw_root={raw_root}")
    print(f"[info] out_root={out_root}")
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)
    process(raw_root, out_root, cfg)


if __name__ == "__main__":
    main()
