# -*- coding: utf-8 -*-
"""批量调用 API 跑评测 prompt 集，每个 provider × 每个 strategy 生成一份 JSON。

用法：
    python models/api_client/run_api.py --config configs/default.yaml \
        --prompts prompts/test_prompts.json \
        --output_dir eval/results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.api_client.prompt_templates import build_messages
from models.api_client.providers import build_provider, chat_with_retry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--prompts", default="prompts/test_prompts.json")
    parser.add_argument("--output_dir", default="eval/results")
    parser.add_argument(
        "--providers",
        default=None,
        help="逗号分隔，覆盖 config 中 enabled 列表，例如：deepseek,zhipu",
    )
    parser.add_argument(
        "--strategies",
        default=None,
        help="逗号分隔，覆盖 config 中 strategies 列表，例如：zero_shot,few_shot,cot",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="只跑前 N 条 prompt（调试用，0=不限）"
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    providers = (
        args.providers.split(",") if args.providers else cfg["api"]["enabled"]
    )
    strategies = (
        args.strategies.split(",") if args.strategies else cfg["api"]["strategies"]
    )

    with open(args.prompts, "r", encoding="utf-8") as f:
        pjson = json.load(f)
    prompts = pjson["prompts"]
    if args.limit > 0:
        prompts = prompts[: args.limit]

    for pname in providers:
        try:
            provider = build_provider(pname, cfg)
        except Exception as e:
            print(f"[skip] provider={pname}: {e}")
            continue
        for strat in strategies:
            results = []
            for p in tqdm(prompts, desc=f"{pname}/{strat}"):
                messages = build_messages(strat, p)
                try:
                    text = chat_with_retry(provider, messages, cfg)
                except Exception as e:
                    print(f"[err] {p['id']}: {e}")
                    text = ""
                results.append(
                    {
                        "id": p["id"],
                        "theme": p.get("theme"),
                        "input_type": p.get("input_type"),
                        "instruction": p.get("instruction"),
                        "format_spec": p.get("format_spec"),
                        "outputs": [text],
                        "model": f"{pname}_{strat}",
                    }
                )
            fname = out_dir / f"{pname}_{strat}.json"
            with fname.open("w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"[ok] {fname} ({len(results)} prompts)")


if __name__ == "__main__":
    main()
