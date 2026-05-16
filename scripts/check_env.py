# -*- coding: utf-8 -*-
"""快速环境自检：是否能跑通核心管线（不实际训练）。"""
from __future__ import annotations

import importlib
import os
import sys

REQUIRED = [
    ("yaml", "PyYAML"),
    ("torch", "torch"),
    ("tqdm", "tqdm"),
    ("opencc", "opencc-python-reimplemented"),
    ("pypinyin", "pypinyin"),
    ("matplotlib", "matplotlib"),
]
OPTIONAL = [
    ("transformers", "transformers"),
    ("peft", "peft"),
    ("openai", "openai"),
    ("zhipuai", "zhipuai"),
    ("dashscope", "dashscope"),
    ("gradio", "gradio"),
    ("wordcloud", "wordcloud"),
]


def check(name: str, pkg: str, required: bool) -> bool:
    try:
        importlib.import_module(name)
        ver = getattr(importlib.import_module(name), "__version__", "?")
        print(f"  [ok] {name}=={ver}")
        return True
    except Exception as e:
        kind = "ERR" if required else "warn"
        print(f"  [{kind}] {pkg} 缺失: {e}")
        return not required


def main() -> None:
    print("Python:", sys.version)
    print("\n[required]")
    ok = True
    for n, p in REQUIRED:
        ok &= check(n, p, required=True)
    print("\n[optional]")
    for n, p in OPTIONAL:
        check(n, p, required=False)

    try:
        import torch

        print(f"\nCUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
    except Exception as e:
        print(f"[warn] torch 信息读取失败: {e}")

    print("\nEnv vars:")
    for k in ("DEEPSEEK_API_KEY", "ZHIPU_API_KEY", "DASHSCOPE_API_KEY"):
        v = os.environ.get(k)
        print(f"  {k} = {'<set>' if v else '<unset>'}")

    print("\nDone." if ok else "\n请安装缺失的必需依赖。")


if __name__ == "__main__":
    main()
