# -*- coding: utf-8 -*-
"""Gradio 演示界面：统一调用 LSTM / API / GPT-2-LoRA。

启动：
    python app/gradio_demo.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import gradio as gr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import evaluate_one
from models.api_client.prompt_templates import build_messages
from models.api_client.providers import build_provider, chat_with_retry


FORMAT_TYPES = ["jueju5", "jueju7", "lvshi5", "lvshi7", "ci"]
FORMAT_LABELS = {
    "jueju5": "五言绝句",
    "jueju7": "七言绝句",
    "lvshi5": "五言律诗",
    "lvshi7": "七言律诗",
    "ci": "词（指定词牌）",
}


def make_prompt_dict(
    topic: str,
    keywords: str,
    first_line: str,
    acrostic: str,
    rhythmic: str,
    style: str,
    fmt_type: str,
):
    """根据界面输入合成一个 prompt 字典。"""
    fmt_spec = {"type": fmt_type}
    cpl_map = {"jueju5": 5, "jueju7": 7, "lvshi5": 5, "lvshi7": 7}
    lines_map = {"jueju5": 4, "jueju7": 4, "lvshi5": 8, "lvshi7": 8}
    if fmt_type in cpl_map:
        fmt_spec["chars_per_line"] = cpl_map[fmt_type]
        fmt_spec["lines"] = lines_map[fmt_type]
    if rhythmic and fmt_type == "ci":
        fmt_spec["rhythmic"] = rhythmic
    if acrostic:
        fmt_spec["acrostic"] = acrostic

    parts = []
    if topic:
        parts.append(f"主题：{topic}")
    if keywords:
        parts.append(f"关键词：{keywords}")
    if first_line:
        parts.append(f"上句：{first_line}")
    if style:
        parts.append(f"风格：{style}")
    if acrostic:
        parts.append(f"藏头：{acrostic}")
    if fmt_type == "ci" and rhythmic:
        parts.append(f"词牌：《{rhythmic}》")

    instr = (
        f"请按要求创作一首{FORMAT_LABELS.get(fmt_type,'诗')}。"
        + ("；".join(parts) if parts else "题材自定。")
    )
    inp = {}
    if topic: inp["topic"] = topic
    if keywords: inp["keywords"] = [k.strip() for k in keywords.split(",") if k.strip()]
    if first_line: inp["first_line"] = first_line
    if acrostic: inp["acrostic_chars"] = acrostic
    if rhythmic: inp["rhythmic"] = rhythmic
    if style: inp["style"] = style
    it = "topic_only"
    if first_line: it = "first_line"
    elif acrostic: it = "acrostic"
    elif rhythmic and fmt_type == "ci": it = "ci_pai"
    elif style: it = "style_imitation"
    elif keywords: it = "topic_keywords"
    return {
        "id": "live",
        "theme": topic or "",
        "input_type": it,
        "input": inp,
        "instruction": instr,
        "format_spec": fmt_spec,
    }


def build_app(cfg: dict):
    # 资源延迟加载
    state = {"lstm": None, "gpt2": None, "providers": {}}

    def _get_lstm():
        if state["lstm"] is None:
            from models.char_lstm.generate import LSTMGenerator

            ck = Path("ckpts/char_lstm")
            if not (ck / "best.pt").exists() and not (ck / "last.pt").exists():
                return None
            # 优先 best.pt
            try:
                state["lstm"] = LSTMGenerator(ck)
            except Exception as e:
                print(f"[warn] 加载 LSTM 失败：{e}")
                return None
        return state["lstm"]

    def _get_gpt2():
        if state["gpt2"] is None:
            ck = Path("ckpts/gpt2_lora/last")
            if not ck.exists():
                return None
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                tok = AutoTokenizer.from_pretrained(ck, trust_remote_code=True)
                try:
                    from peft import PeftModel

                    base = AutoModelForCausalLM.from_pretrained(
                        cfg["gpt2"]["base_model"], trust_remote_code=True
                    )
                    model = PeftModel.from_pretrained(base, ck)
                except Exception:
                    model = AutoModelForCausalLM.from_pretrained(ck, trust_remote_code=True)
                import torch as _torch

                device = "cuda" if _torch.cuda.is_available() else "cpu"
                model.to(device).eval()
                state["gpt2"] = (tok, model, device)
            except Exception as e:
                print(f"[warn] 加载 GPT-2 失败：{e}")
                return None
        return state["gpt2"]

    def _get_provider(name: str):
        if name not in state["providers"]:
            try:
                state["providers"][name] = build_provider(name, cfg)
            except Exception as e:
                print(f"[warn] {name}: {e}")
                state["providers"][name] = None
        return state["providers"][name]

    def generate(method, strategy, fmt_type, topic, keywords, first_line, acrostic, rhythmic, style, temperature, top_k):
        pdict = make_prompt_dict(topic, keywords, first_line, acrostic, rhythmic, style, fmt_type)
        if method == "Char-LSTM (本地)":
            gen = _get_lstm()
            if gen is None:
                return "未找到 ckpts/char_lstm/，请先运行训练脚本。", ""
            from models.char_lstm.generate import run_prompt

            text = run_prompt(gen, pdict, temperature=float(temperature), top_k=int(top_k))
        elif method == "GPT-2-LoRA (本地)":
            g = _get_gpt2()
            if g is None:
                return "未找到 ckpts/gpt2_lora/last，请先运行 train_lora.py。", ""
            tok, model, device = g
            from models.gpt2_lora.generate import build_prefix
            import torch as _torch

            prefix = build_prefix(pdict)
            ids = tok.encode(prefix, return_tensors="pt").to(device)
            with _torch.no_grad():
                out = model.generate(
                    input_ids=ids,
                    max_new_tokens=80,
                    do_sample=True,
                    temperature=float(temperature),
                    top_k=int(top_k),
                    top_p=0.9,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            text = tok.decode(out[0], skip_special_tokens=True)
            for tag in ["[五绝]", "[七绝]", "[五律]", "[七律]", "[词]"]:
                text = text.replace(tag, "")
            text = text.strip()
        else:
            # API
            name_map = {"DeepSeek": "deepseek", "智谱 GLM": "zhipu", "通义千问": "qwen"}
            pname = name_map.get(method)
            if pname is None:
                return f"未知方法：{method}", ""
            provider = _get_provider(pname)
            if provider is None:
                return f"未配置 {method} API（请设置环境变量）。", ""
            msgs = build_messages(strategy or "zero_shot", pdict)
            try:
                text = chat_with_retry(provider, msgs, cfg, temperature=float(temperature))
            except Exception as e:
                return f"API 调用失败：{e}", ""

        # 计算指标
        m = evaluate_one(text, pdict["format_spec"])
        info = (
            f"格式合规：{m['format_acc']:.2f} | 押韵率：{m['rhyme_acc']:.2f} | "
            f"2-gram重复：{m['repetition_2']:.2f} | 字数：{int(m['len'])}"
        )
        return text, info

    with gr.Blocks(title="吟诗作赋演示") as demo:
        gr.Markdown("# 吟诗作赋 · 多模型对比演示")
        gr.Markdown(
            "支持四种方法：**Char-LSTM 本地基线** / **API：DeepSeek / 智谱 / 通义** / **GPT-2-LoRA**。\n"
            "输入主题、关键词、上句、藏头字等之一即可。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                method = gr.Radio(
                    ["Char-LSTM (本地)", "DeepSeek", "智谱 GLM", "通义千问", "GPT-2-LoRA (本地)"],
                    value="Char-LSTM (本地)",
                    label="生成方法",
                )
                strategy = gr.Radio(
                    ["zero_shot", "few_shot", "cot"],
                    value="zero_shot",
                    label="API Prompt 策略",
                )
                fmt = gr.Radio(
                    list(FORMAT_LABELS.keys()),
                    value="jueju7",
                    label="诗体（jueju5=五绝，jueju7=七绝，lvshi5=五律，lvshi7=七律，ci=词）",
                )
                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
                top_k = gr.Slider(1, 100, value=8, step=1, label="Top-K")
            with gr.Column(scale=2):
                topic = gr.Textbox(label="主题（例如：春雨 / 边塞 / 月夜）")
                keywords = gr.Textbox(label="关键词（逗号分隔）")
                first_line = gr.Textbox(label="上句（续写时填）")
                acrostic = gr.Textbox(label="藏头字（每句首字）")
                rhythmic = gr.Textbox(label="词牌（仅 ci 模式）")
                style = gr.Textbox(label="风格（例如：李白 / 杜甫 / 婉约）")
                btn = gr.Button("生成", variant="primary")
                out_text = gr.Textbox(label="生成结果", lines=6)
                out_info = gr.Markdown()
        btn.click(
            generate,
            inputs=[method, strategy, fmt, topic, keywords, first_line, acrostic, rhythmic, style, temperature, top_k],
            outputs=[out_text, out_info],
        )

        gr.Examples(
            examples=[
                ["DeepSeek", "zero_shot", "jueju7", "春雨", "细雨,桃花,燕子", "", "", "", "", 0.8, 8],
                ["DeepSeek", "few_shot", "jueju5", "送别", "长亭,杨柳", "", "", "", "王维", 0.8, 8],
                ["DeepSeek", "cot", "lvshi7", "登高怀古", "", "", "", "", "", 0.7, 8],
                ["Char-LSTM (本地)", "zero_shot", "jueju7", "", "", "", "春雨绵绵", "", "", 0.9, 8],
                ["智谱 GLM", "zero_shot", "ci", "中秋望月", "", "", "", "水调歌头", "", 0.8, 50],
            ],
            inputs=[method, strategy, fmt, topic, keywords, first_line, acrostic, rhythmic, style, temperature, top_k],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    demo = build_app(cfg)
    gcfg = cfg.get("gradio", {})
    demo.launch(
        server_name=gcfg.get("server_name", "127.0.0.1"),
        server_port=gcfg.get("server_port", 7860),
        share=gcfg.get("share", False),
    )


if __name__ == "__main__":
    main()
