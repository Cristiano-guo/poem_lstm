# -*- coding: utf-8 -*-
"""统一封装多个 API 提供商：DeepSeek、智谱 GLM、通义千问。

外部接口：
    provider = build_provider(name, cfg)
    text = provider.chat(messages, **gen_kwargs)
"""
from __future__ import annotations

import os
import time
from typing import Dict, List

# ----------------------------------------------------------
# 基础类
# ----------------------------------------------------------
class BaseProvider:
    name: str = "base"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        raise NotImplementedError


# ----------------------------------------------------------
# DeepSeek（OpenAI 兼容协议）
# ----------------------------------------------------------
class DeepSeekProvider(BaseProvider):
    name = "deepseek"

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("请安装 openai: pip install openai") from e

        api_key_env = cfg["api"]["env_keys"]["deepseek"]
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"环境变量 {api_key_env} 未设置")
        base_url = cfg["api"]["endpoints"]["deepseek"]
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = cfg["api"]["models"]["deepseek"]

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        temperature = kwargs.get("temperature", self.cfg["api"]["temperature"])
        max_tokens = kwargs.get("max_tokens", self.cfg["api"]["max_tokens"])
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


# ----------------------------------------------------------
# 智谱 GLM
# ----------------------------------------------------------
class ZhipuProvider(BaseProvider):
    name = "zhipu"

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        try:
            from zhipuai import ZhipuAI
        except ImportError as e:
            raise ImportError("请安装 zhipuai: pip install zhipuai") from e
        api_key_env = cfg["api"]["env_keys"]["zhipu"]
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"环境变量 {api_key_env} 未设置")
        self.client = ZhipuAI(api_key=api_key)
        self.model = cfg["api"]["models"]["zhipu"]

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        temperature = kwargs.get("temperature", self.cfg["api"]["temperature"])
        max_tokens = kwargs.get("max_tokens", self.cfg["api"]["max_tokens"])
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


# ----------------------------------------------------------
# 通义千问（dashscope）
# ----------------------------------------------------------
class QwenProvider(BaseProvider):
    name = "qwen"

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        try:
            import dashscope  # noqa: F401
        except ImportError as e:
            raise ImportError("请安装 dashscope: pip install dashscope") from e
        api_key_env = cfg["api"]["env_keys"]["qwen"]
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"环境变量 {api_key_env} 未设置")
        import dashscope

        dashscope.api_key = api_key
        self.model = cfg["api"]["models"]["qwen"]
        self.dashscope = dashscope

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        temperature = kwargs.get("temperature", self.cfg["api"]["temperature"])
        max_tokens = kwargs.get("max_tokens", self.cfg["api"]["max_tokens"])
        from dashscope import Generation

        resp = Generation.call(
            model=self.model,
            messages=messages,
            result_format="message",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"qwen api error: {resp.code} {resp.message}")
        return resp.output.choices[0].message.content or ""


# ----------------------------------------------------------
# 工厂
# ----------------------------------------------------------
PROVIDERS = {
    "deepseek": DeepSeekProvider,
    "zhipu": ZhipuProvider,
    "qwen": QwenProvider,
}


def build_provider(name: str, cfg: dict) -> BaseProvider:
    if name not in PROVIDERS:
        raise ValueError(f"unknown provider: {name}")
    return PROVIDERS[name](cfg)


def chat_with_retry(provider: BaseProvider, messages, cfg: dict, **kwargs) -> str:
    n_retry = cfg["api"].get("retry", 3)
    wait = cfg["api"].get("retry_wait", 2)
    last_err: Exception | None = None
    for i in range(n_retry):
        try:
            return provider.chat(messages, **kwargs)
        except Exception as e:
            last_err = e
            print(f"[warn] provider={provider.name} 第 {i+1} 次失败: {e}")
            time.sleep(wait * (i + 1))
    raise RuntimeError(f"调用 {provider.name} 连续失败：{last_err}")
