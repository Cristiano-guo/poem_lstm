# -*- coding: utf-8 -*-
"""使用 GPT-2-LoRA 检查点对评测 prompt 集做生成。

把每条 prompt 渲染成"<体裁> + 提示前缀"作为续写起点。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


FMT_TAG = {
    "jueju5": "[五绝]",
    "jueju7": "[七绝]",
    "lvshi5": "[五律]",
    "lvshi7": "[七律]",
    "ci": "[词]",
}


def build_prefix(p: dict) -> str:
    fmt = p.get("format_spec", {}) or {}
    ft = fmt.get("type", "jueju7")
    tag = FMT_TAG.get(ft, "[七绝]")
    it = p.get("input_type", "topic_only")
    inp = p.get("input", {})
    if it == "first_line":
        return f"{tag}{inp.get('first_line','')}，"
    if it == "ci_pai":
        return f"{tag}《{fmt.get('rhythmic','')}》"
    if it == "acrostic":
        return tag
    # 其他情况：把主题词作为标题
    title = inp.get("topic") or "无题"
    return f"{tag}《{title}》"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt_dir", default="ckpts/gpt2_lora/last")
    parser.add_argument("--prompts", default="prompts/test_prompts.json")
    parser.add_argument("--output", default="eval/results/gpt2_lora.json")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = cfg["gpt2"]["base_model"]
    tok = AutoTokenizer.from_pretrained(args.ckpt_dir, trust_remote_code=True)

    # 优先按 peft adapter 加载
    try:
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, args.ckpt_dir)
        print("[info] loaded peft adapter")
    except Exception as e:
        print(f"[warn] peft 加载失败 ({e})，回退到 ckpt_dir 整模型")
        model = AutoModelForCausalLM.from_pretrained(args.ckpt_dir, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    with open(args.prompts, "r", encoding="utf-8") as f:
        pjson = json.load(f)
    prompts = pjson["prompts"]

    results = []
    for p in tqdm(prompts, desc="gpt2_lora"):
        prefix = build_prefix(p)
        ids = tok.encode(prefix, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        # 去掉 prefix 中的标签
        for t in FMT_TAG.values():
            text = text.replace(t, "")
        text = text.strip()
        results.append(
            {
                "id": p["id"],
                "theme": p.get("theme"),
                "input_type": p.get("input_type"),
                "instruction": p.get("instruction"),
                "format_spec": p.get("format_spec"),
                "outputs": [text],
                "model": "gpt2_lora",
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[done] -> {out_path}")


if __name__ == "__main__":
    main()
