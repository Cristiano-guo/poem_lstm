# -*- coding: utf-8 -*-
"""扫描 eval/results/*.json，按方法计算各指标，输出 summary.csv 与 detail.json。

用法：
    python eval/auto_metrics.py --results_dir eval/results --refs data/processed
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import aggregate, evaluate_one


# 从训练集随机抽样作为 BLEU/ROUGE 的参考池（与 prompt 同主题尽量匹配——这里简化为随机）
def build_ref_pool(processed_dir: Path, n_per_fmt: int = 200) -> Dict[str, List[str]]:
    name_map = {
        "jueju5": "tang_jueju5.txt",
        "jueju7": "tang_jueju7.txt",
        "lvshi5": "tang_lvshi5.txt",
        "lvshi7": "tang_lvshi7.txt",
        "ci": "ci_song.txt",
    }
    pool: Dict[str, List[str]] = {}
    for fmt, fn in name_map.items():
        p = processed_dir / fn
        if not p.exists():
            pool[fmt] = []
            continue
        lines = p.read_text(encoding="utf-8").splitlines()
        random.shuffle(lines)
        # 词文件里是 "<词牌>\t<正文>"
        if fmt == "ci":
            lines = [l.split("\t", 1)[-1] for l in lines if l.strip()]
        pool[fmt] = lines[:n_per_fmt]
    return pool


def evaluate_file(path: Path, ref_pool: Dict[str, List[str]]) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    per_sample = []
    details = []
    for it in items:
        spec = it.get("format_spec", {}) or {}
        ft = spec.get("type", "jueju7")
        refs = ref_pool.get(ft, [])[:50]  # 用前 50 条作为参考
        outputs = it.get("outputs") or [""]
        text = outputs[0] if outputs else ""
        s = evaluate_one(text, spec, refs=refs)
        per_sample.append(s)
        details.append({"id": it["id"], "model": it.get("model"), "text": text, **s})
    summary = aggregate(per_sample)
    return {"summary": summary, "details": details}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="eval/results")
    parser.add_argument("--refs", default="data/processed")
    parser.add_argument("--out_csv", default="eval/results/summary.csv")
    parser.add_argument("--out_detail", default="eval/results/detail.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    rdir = Path(args.results_dir)
    refs = build_ref_pool(Path(args.refs))

    rows = []
    all_details = []
    for p in sorted(rdir.glob("*.json")):
        if p.name in ("summary.json", "detail.json"):
            continue
        try:
            r = evaluate_file(p, refs)
        except Exception as e:
            print(f"[skip] {p.name}: {e}")
            continue
        s = r["summary"]
        s["model"] = p.stem
        s["n"] = len(r["details"])
        rows.append(s)
        all_details.extend(r["details"])
        print(f"[ok] {p.name}: {s}")

    if not rows:
        print("[warn] 没有可评估的结果文件")
        return

    # 输出 CSV
    keys = sorted({k for r in rows for k in r.keys()})
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[done] summary -> {out_csv}")

    with open(args.out_detail, "w", encoding="utf-8") as f:
        json.dump(all_details, f, ensure_ascii=False, indent=2)
    print(f"[done] detail -> {args.out_detail}")


if __name__ == "__main__":
    main()
