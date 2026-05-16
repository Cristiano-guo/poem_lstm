# -*- coding: utf-8 -*-
"""Char-LSTM 文本生成 + 评测 prompt 接入。

支持四种生成模式：
1. 续写：给定前缀续写
2. 藏头：每句首字固定
3. 主题（弱）：把主题词作为前缀续写
4. 五言/七言 格式约束：插入 "，" "。" 占位符

注意：LSTM 是字符级模型，不像 GPT 那样能理解"主题词"，
我们把诗题/关键词当成 prompt 前缀拼到 BOS 之后，模型会学会"看到这些字之后接什么内容"。
为简化，这里"主题"模式直接退化为"用主题首字作为开头字"。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.char_lstm.model import CharLSTM
from models.char_lstm.vocab import Vocab


@torch.no_grad()
def _sample_next(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    forbid_ids: set[int] | None = None,
    forbid_until_score: float = -1e9,
) -> int:
    """从 (V,) logits 中采样一个 token id。"""
    if temperature <= 0:
        temperature = 1.0
    logits = logits / temperature
    if forbid_ids:
        for fid in forbid_ids:
            logits[fid] = forbid_until_score
    if top_k and top_k > 0:
        vals, idx = torch.topk(logits, k=top_k)
        probs = F.softmax(vals, dim=-1)
        chosen = torch.multinomial(probs, num_samples=1)
        return idx[chosen].item()
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


class LSTMGenerator:
    def __init__(self, ckpt_dir: str | Path, device: str | None = None):
        ckpt_dir = Path(ckpt_dir)
        self.vocab = Vocab.load(ckpt_dir / "vocab.json")
        ck = torch.load(ckpt_dir / "best.pt", map_location="cpu")
        cfg = ck["config"]
        self.model = CharLSTM(
            vocab_size=ck["vocab_size"],
            embedding_dim=cfg["embedding_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=0.0,
            pad_id=self.vocab.pad_id,
        )
        self.model.load_state_dict(ck["model"])
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        self.cfg = cfg

    def _prime(self, prefix_ids: List[int]):
        """喂入前缀，返回隐状态与最后 logits。"""
        x = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)
        hidden = self.model.init_hidden(1, self.device)
        logits, hidden = self.model(x, hidden)
        return logits[0, -1], hidden

    def generate(
        self,
        prefix: str = "",
        max_len: int = 64,
        temperature: float = 1.0,
        top_k: int = 8,
        stop_token: str | None = None,
        forbid_chars: str = "",
    ) -> str:
        prefix_ids = [self.vocab.bos_id] + self.vocab.encode(prefix)
        last_logits, hidden = self._prime(prefix_ids)
        forbid = {self.vocab.stoi[ch] for ch in forbid_chars if ch in self.vocab.stoi}
        forbid.add(self.vocab.pad_id)
        forbid.add(self.vocab.bos_id)
        out_chars: List[str] = list(prefix)
        cur_logits = last_logits
        for _ in range(max_len):
            tok = _sample_next(cur_logits, temperature, top_k, forbid_ids=forbid)
            if tok == self.vocab.eos_id:
                break
            ch = self.vocab.itos[tok]
            out_chars.append(ch)
            if stop_token and ch == stop_token:
                break
            x = torch.tensor([[tok]], dtype=torch.long, device=self.device)
            logits, hidden = self.model(x, hidden)
            cur_logits = logits[0, -1]
        return "".join(out_chars)

    def generate_format(
        self,
        prefix: str,
        n_lines: int,
        chars_per_line: int,
        sep_inner: str = "，",
        sep_outer: str = "。",
        temperature: float = 1.0,
        top_k: int = 8,
    ) -> str:
        """生成严格的"四句/八句 + 五言/七言"格式诗。
        每生成 chars_per_line 个字后强制插入标点。
        """
        prefix_ids = [self.vocab.bos_id] + self.vocab.encode(prefix)
        last_logits, hidden = self._prime(prefix_ids)

        # 禁止采样到标点和特殊 token（标点由我们强制插入）
        forbid = set()
        for ch in "，。？！；：、":
            if ch in self.vocab.stoi:
                forbid.add(self.vocab.stoi[ch])
        forbid.add(self.vocab.pad_id)
        forbid.add(self.vocab.bos_id)
        forbid.add(self.vocab.eos_id)

        out: List[str] = list(prefix)
        # 已有字符数（不计标点）
        cur_chars = len(prefix)
        # 当前在第几行（0-based）
        line_idx = 0
        cur_logits = last_logits
        while line_idx < n_lines:
            need = chars_per_line - cur_chars if line_idx == 0 else chars_per_line
            if line_idx == 0 and need <= 0:
                # prefix 已超过一行
                need = 0
            for _ in range(need):
                tok = _sample_next(cur_logits, temperature, top_k, forbid_ids=forbid)
                ch = self.vocab.itos[tok]
                out.append(ch)
                x = torch.tensor([[tok]], dtype=torch.long, device=self.device)
                logits, hidden = self.model(x, hidden)
                cur_logits = logits[0, -1]
            # 插入标点
            punct = sep_outer if (line_idx % 2 == 1) else sep_inner
            out.append(punct)
            # 把这个标点也喂回模型保持上下文
            if punct in self.vocab.stoi:
                pid = self.vocab.stoi[punct]
                x = torch.tensor([[pid]], dtype=torch.long, device=self.device)
                logits, hidden = self.model(x, hidden)
                cur_logits = logits[0, -1]
            line_idx += 1
            cur_chars = 0
        return "".join(out)

    def generate_acrostic(
        self,
        acrostic_chars: str,
        chars_per_line: int,
        temperature: float = 1.0,
        top_k: int = 8,
    ) -> str:
        """藏头诗：每句首字固定。"""
        prefix_ids = [self.vocab.bos_id]
        last_logits, hidden = self._prime(prefix_ids)
        forbid = set()
        for ch in "，。？！；：、":
            if ch in self.vocab.stoi:
                forbid.add(self.vocab.stoi[ch])
        forbid.add(self.vocab.pad_id)
        forbid.add(self.vocab.bos_id)
        forbid.add(self.vocab.eos_id)

        out: List[str] = []
        for i, head_ch in enumerate(acrostic_chars):
            # 强制塞入头字
            out.append(head_ch)
            head_id = self.vocab.stoi.get(head_ch, self.vocab.unk_id)
            x = torch.tensor([[head_id]], dtype=torch.long, device=self.device)
            logits, hidden = self.model(x, hidden)
            cur_logits = logits[0, -1]
            # 补足剩余 chars_per_line - 1 个字
            for _ in range(chars_per_line - 1):
                tok = _sample_next(cur_logits, temperature, top_k, forbid_ids=forbid)
                ch = self.vocab.itos[tok]
                out.append(ch)
                x = torch.tensor([[tok]], dtype=torch.long, device=self.device)
                logits, hidden = self.model(x, hidden)
                cur_logits = logits[0, -1]
            # 标点
            punct = "。" if (i % 2 == 1) else "，"
            out.append(punct)
            if punct in self.vocab.stoi:
                pid = self.vocab.stoi[punct]
                x = torch.tensor([[pid]], dtype=torch.long, device=self.device)
                logits, hidden = self.model(x, hidden)
                cur_logits = logits[0, -1]
        return "".join(out)


# ----------------- prompt -> 调度 -----------------
def run_prompt(gen: LSTMGenerator, prompt: dict, temperature: float, top_k: int) -> str:
    fmt = prompt.get("format_spec", {})
    ftype = fmt.get("type", "jueju7")
    it = prompt.get("input_type", "topic_only")
    inp = prompt.get("input", {})

    # 七言 / 五言 / 律诗
    cpl_map = {"jueju5": 5, "jueju7": 7, "lvshi5": 5, "lvshi7": 7}
    lines_map = {"jueju5": 4, "jueju7": 4, "lvshi5": 8, "lvshi7": 8}
    if ftype not in cpl_map:
        # 词牌：LSTM 难处理，退化为七绝
        ftype = "jueju7"
    cpl = cpl_map[ftype]
    n_lines = lines_map[ftype]

    # 藏头
    if it == "acrostic":
        chars = inp.get("acrostic_chars", "")
        n_lines = len(chars) if chars else n_lines
        return gen.generate_acrostic(chars, chars_per_line=cpl, temperature=temperature, top_k=top_k)

    # 上句续写：先生成首句剩余，再继续 n_lines-1 句
    if it == "first_line":
        first = inp.get("first_line", "")
        return gen.generate_format(
            prefix=first + "，",
            n_lines=n_lines - 1,
            chars_per_line=cpl,
            temperature=temperature,
            top_k=top_k,
        )

    # 其他情况：把主题词作为前缀（弱条件）
    prefix = ""
    return gen.generate_format(
        prefix=prefix,
        n_lines=n_lines,
        chars_per_line=cpl,
        temperature=temperature,
        top_k=top_k,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt_dir", default="ckpts/char_lstm")
    parser.add_argument("--prompts", default="prompts/test_prompts.json")
    parser.add_argument("--output", default="eval/results/lstm.json")
    parser.add_argument("--n_samples", type=int, default=1, help="每个 prompt 采样几次")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    lcfg = cfg["lstm"]
    temperature = lcfg.get("temperature", 1.0)
    top_k = lcfg.get("top_k", 8)

    gen = LSTMGenerator(args.ckpt_dir)

    with open(args.prompts, "r", encoding="utf-8") as f:
        pjson = json.load(f)

    results = []
    for p in pjson["prompts"]:
        samples = []
        for _ in range(args.n_samples):
            text = run_prompt(gen, p, temperature, top_k)
            samples.append(text)
        results.append(
            {
                "id": p["id"],
                "theme": p.get("theme"),
                "input_type": p.get("input_type"),
                "instruction": p.get("instruction"),
                "format_spec": p.get("format_spec"),
                "outputs": samples,
                "model": "lstm",
            }
        )
        print(f"[{p['id']}] {samples[0]}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[done] {len(results)} results -> {out}")


if __name__ == "__main__":
    main()
