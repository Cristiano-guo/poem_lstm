# -*- coding: utf-8 -*-
"""根据 eval/results 下的各方法 JSON，生成可填写的人工评价 CSV。

每条 prompt × 每个模型一行，方便随机抽样 10 条评分。
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="eval/results")
    parser.add_argument("--out_csv", default="eval/results/human_eval.csv")
    parser.add_argument("--n_prompts", type=int, default=10, help="随机抽样多少条 prompt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    rdir = Path(args.results_dir)
    all_files = sorted(p for p in rdir.glob("*.json") if p.stem not in {"summary", "detail"})
    if not all_files:
        print("[warn] 无结果文件")
        return

    # 用第一个文件确定 prompt 顺序
    with all_files[0].open("r", encoding="utf-8") as f:
        first = json.load(f)
    prompt_ids = [it["id"] for it in first]
    sample_ids = random.sample(prompt_ids, k=min(args.n_prompts, len(prompt_ids)))
    print(f"[info] 抽样 prompt: {sample_ids}")

    rows = []
    for fp in all_files:
        with fp.open("r", encoding="utf-8") as f:
            items = json.load(f)
        model_name = fp.stem
        id2text = {it["id"]: (it.get("outputs") or [""])[0] for it in items}
        id2meta = {it["id"]: it for it in items}
        for pid in sample_ids:
            meta = id2meta.get(pid, {})
            rows.append(
                {
                    "prompt_id": pid,
                    "theme": meta.get("theme", ""),
                    "input_type": meta.get("input_type", ""),
                    "instruction": meta.get("instruction", ""),
                    "model": model_name,
                    "output": id2text.get(pid, ""),
                    "fluency_1to5": "",
                    "relevance_1to5": "",
                    "format_1to5": "",
                    "style_1to5": "",
                    "poetic_1to5": "",
                    "note": "",
                }
            )

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[done] {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
