# -*- coding: utf-8 -*-
"""GPT-2-Chinese LoRA 微调（可在 4060 8GB 上跑）。

用法：
    python models/gpt2_lora/train_lora.py --config configs/default.yaml

依赖：transformers, peft, accelerate
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.gpt2_lora.dataset import PoemSFTDataset, collate_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out_dir", default="ckpts/gpt2_lora")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gcfg = cfg["gpt2"]
    paths = cfg["paths"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 延迟导入，方便 LSTM-only 环境也能用本项目
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = gcfg["base_model"]
    print(f"[info] loading base model: {base}")
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.sep_token
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.float16 if gcfg.get("fp16", True) else torch.float32,
        trust_remote_code=True,
    )

    # LoRA
    if gcfg.get("use_lora", True):
        from peft import LoraConfig, TaskType, get_peft_model

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=gcfg["lora_r"],
            lora_alpha=gcfg["lora_alpha"],
            lora_dropout=gcfg["lora_dropout"],
            target_modules=gcfg.get("target_modules", ["c_attn"]),
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if gcfg.get("fp16", True) and device.type == "cuda":
        model = model.half()  # 简化处理；正式工程建议 amp autocast
    print(f"[info] device = {device}")

    # 数据
    processed = Path(paths["processed_root"])
    files = [processed / fn.replace(".txt", ".train.txt") for fn in gcfg["train_files"]]
    ds = PoemSFTDataset(files, tok, max_length=gcfg["max_length"])
    loader = DataLoader(
        ds,
        batch_size=gcfg["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=partial(collate_fn, pad_id=tok.pad_token_id),
        drop_last=True,
    )

    # 优化器
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=gcfg["lr"]
    )

    history = {"train_loss": []}
    grad_accum = gcfg.get("grad_accum", 1)

    for epoch in range(1, gcfg["epochs"] + 1):
        model.train()
        ep_loss = 0.0
        n_step = 0
        t0 = time.time()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        optim.zero_grad()
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0:
                optim.step()
                optim.zero_grad()
            ep_loss += out.loss.item()
            n_step += 1
            pbar.set_postfix(loss=f"{out.loss.item():.4f}")
        avg = ep_loss / max(1, n_step)
        history["train_loss"].append(avg)
        print(f"[epoch {epoch}] avg_loss={avg:.4f} ppl~{math.exp(avg):.2f} time={time.time()-t0:.1f}s")

        # 保存：LoRA adapter
        save_dir = out_dir / f"epoch_{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            model.save_pretrained(save_dir)  # peft 会只存 adapter
        except Exception:
            torch.save(model.state_dict(), save_dir / "pytorch_model.bin")
        tok.save_pretrained(save_dir)

    # 最终也保存到 last/
    last_dir = out_dir / "last"
    last_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(last_dir)
    except Exception:
        torch.save(model.state_dict(), last_dir / "pytorch_model.bin")
    tok.save_pretrained(last_dir)
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
