# -*- coding: utf-8 -*-
"""中文古典诗词生成评估指标：
- format_acc      格式合规率（句数 / 字数）
- rhyme_acc       押韵率（偶数句末字普通话韵母一致）
- repetition      n-gram 重复率（越低越好）
- distinct_2      2-gram 多样性
- bleu_2 / bleu_4 与参考诗的 BLEU
- rouge_l         与参考诗的 ROUGE-L F
- avg_len         平均字符数
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Sequence

try:
    from pypinyin import Style, lazy_pinyin
except ImportError:
    lazy_pinyin = None

# ----------------------------------------------------------
# 工具
# ----------------------------------------------------------
_HAN = re.compile(r"[\u4e00-\u9fff]")
_PUNCT = "，。？！；：、,.!?;:"

_VOWEL_GROUPS = {
    # 把相近韵母合并为同一组（普通话宽韵）
    "a": "a",
    "ai": "ai",
    "ao": "ao",
    "an": "an",
    "ang": "ang",
    "e": "e",
    "ei": "ei",
    "en": "en",
    "eng": "eng",
    "er": "er",
    "i": "i",
    "ia": "ia",
    "iao": "ao",
    "ian": "an",
    "iang": "ang",
    "ie": "e",
    "in": "en",
    "ing": "eng",
    "iong": "ong",
    "o": "o",
    "ou": "ou",
    "ong": "ong",
    "u": "u",
    "ua": "a",
    "uai": "ai",
    "uan": "an",
    "uang": "ang",
    "ue": "e",
    "ui": "ei",
    "un": "en",
    "uo": "o",
    "v": "u",
    "ve": "e",
    "van": "an",
    "vn": "en",
}


def _final(ch: str) -> str:
    if lazy_pinyin is None:
        return ""
    py = lazy_pinyin(ch, style=Style.FINALS)
    if not py:
        return ""
    f = py[0]
    f = re.sub(r"[1-5]$", "", f)
    return _VOWEL_GROUPS.get(f, f)


# ----------------------------------------------------------
# 拆句
# ----------------------------------------------------------
def split_lines(text: str) -> List[str]:
    """按句末标点切分，去掉空句。"""
    text = text.strip()
    # 把多种句末标点统一
    text = re.sub(r"[。？！]", "。", text)
    text = re.sub(r"[，；]", "，", text)
    parts = re.split(r"[，。]", text)
    return [p for p in parts if p.strip()]


def chars_only(line: str) -> str:
    return "".join(ch for ch in line if _HAN.match(ch))


# ----------------------------------------------------------
# 单指标
# ----------------------------------------------------------
def format_acc(text: str, spec: dict) -> float:
    """格式合规：句数 + 每句字数（仅汉字）。

    返回 1.0 / 0.0；若 spec 缺字段返回 0.5（部分合规）。
    """
    if not spec:
        return 0.5
    t = spec.get("type")
    if t == "ci":
        # 词牌：只判断不是空、字符合理
        return 1.0 if len(chars_only(text)) >= 10 else 0.0
    n_lines = spec.get("lines")
    cpl = spec.get("chars_per_line")
    if not n_lines or not cpl:
        return 0.5
    lines = split_lines(text)
    if len(lines) != n_lines:
        return 0.0
    for ln in lines:
        if len(chars_only(ln)) != cpl:
            return 0.0
    # 藏头
    ac = spec.get("acrostic")
    if ac:
        for ch_expect, line in zip(ac, lines):
            if not line.startswith(ch_expect):
                return 0.5
    return 1.0


def rhyme_acc(text: str, spec: dict) -> float:
    """押韵率：取所有偶数句（2、4、…）末字，韵母同组比例。"""
    if not spec or spec.get("type") == "ci":
        return 0.0  # 词牌押韵判定复杂，先跳过
    lines = split_lines(text)
    even_finals = []
    for i, ln in enumerate(lines, start=1):
        if i % 2 != 0:
            continue
        cs = chars_only(ln)
        if not cs:
            continue
        f = _final(cs[-1])
        if f:
            even_finals.append(f)
    if len(even_finals) < 2:
        return 0.0
    most_common, n = Counter(even_finals).most_common(1)[0]
    return n / len(even_finals)


def ngram_repetition(text: str, n: int = 2) -> float:
    """n-gram 重复率：重复次数 / 总 n-gram。"""
    cs = chars_only(text)
    if len(cs) < n:
        return 0.0
    grams = [cs[i : i + n] for i in range(len(cs) - n + 1)]
    cnt = Counter(grams)
    rep = sum(c - 1 for c in cnt.values() if c > 1)
    return rep / len(grams) if grams else 0.0


def distinct_n(text: str, n: int = 2) -> float:
    cs = chars_only(text)
    if len(cs) < n:
        return 0.0
    grams = [cs[i : i + n] for i in range(len(cs) - n + 1)]
    return len(set(grams)) / len(grams)


# ----------------------------------------------------------
# BLEU / ROUGE（与参考池对比）
# ----------------------------------------------------------
def _ngrams(seq: Sequence[str], n: int):
    return Counter(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))


def bleu_n(hyp: str, refs: List[str], max_n: int = 4) -> float:
    """非常简化版的 BLEU（无 brevity penalty 时近似），范围 [0,1]。"""
    hyp_chars = list(chars_only(hyp))
    if not hyp_chars:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        hyp_grams = _ngrams(hyp_chars, n)
        if not hyp_grams:
            return 0.0
        max_ref = Counter()
        for r in refs:
            rg = _ngrams(list(chars_only(r)), n)
            for k, v in rg.items():
                if v > max_ref[k]:
                    max_ref[k] = v
        overlap = 0
        for k, v in hyp_grams.items():
            overlap += min(v, max_ref.get(k, 0))
        precisions.append(overlap / sum(hyp_grams.values()))
    # 几何平均
    if any(p == 0 for p in precisions):
        return 0.0
    import math

    score = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    # brevity penalty
    rl = min(len(list(chars_only(r))) for r in refs) if refs else len(hyp_chars)
    if len(hyp_chars) < rl:
        score *= math.exp(1 - rl / max(1, len(hyp_chars)))
    return float(score)


def rouge_l(hyp: str, refs: List[str]) -> float:
    """ROUGE-L F1（与最相似的参考取 max）。"""
    hyp_chars = list(chars_only(hyp))
    if not hyp_chars:
        return 0.0
    best = 0.0
    for r in refs:
        r_chars = list(chars_only(r))
        if not r_chars:
            continue
        # LCS
        m, n = len(hyp_chars), len(r_chars)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if hyp_chars[i] == r_chars[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
        lcs = dp[m][n]
        if lcs == 0:
            continue
        p = lcs / m
        rec = lcs / n
        f1 = 2 * p * rec / (p + rec)
        if f1 > best:
            best = f1
    return best


# ----------------------------------------------------------
# 汇总
# ----------------------------------------------------------
def evaluate_one(text: str, spec: dict, refs: List[str] | None = None) -> Dict[str, float]:
    refs = refs or []
    out = {
        "format_acc": format_acc(text, spec),
        "rhyme_acc": rhyme_acc(text, spec),
        "repetition_2": ngram_repetition(text, 2),
        "distinct_2": distinct_n(text, 2),
        "len": float(len(chars_only(text))),
    }
    if refs:
        out["bleu_4"] = bleu_n(text, refs, max_n=4)
        out["bleu_2"] = bleu_n(text, refs, max_n=2)
        out["rouge_l"] = rouge_l(text, refs)
    return out


def aggregate(per_sample: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_sample:
        return {}
    keys = set()
    for d in per_sample:
        keys.update(d.keys())
    agg = {}
    for k in keys:
        vals = [d[k] for d in per_sample if k in d]
        agg[k] = sum(vals) / len(vals) if vals else 0.0
    return agg
