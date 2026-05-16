# 吟诗作赋（Poem Generation）实验

> 计算智能上机实验 · 文本生成方向
>
> 目标层级：**较好完成**（≥2 种方法对比 + 定量指标 + 人工评价 + 可视化 + 失败案例分析）

---

## 1. 项目概述

本实验基于 [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) 开源诗词语料库，
完成"根据主题 / 关键词 / 上句 / 风格 / 藏头字" 等输入生成中文古典诗词的任务。

实验对比三类方法：

| 编号 | 方法 | 角色 | 显存需求 |
|---|---|---|---|
| **M1** | Char-LSTM 从零训练 | 本地基线，体现自实现工作 | <2 GB |
| **M2** | 大模型 API + Prompt 工程（zero-shot / few-shot / CoT） | 主力效果 | 0 |
| **M3** | GPT-2-Chinese LoRA 微调（可选） | 进阶加分 | 4~6 GB |

硬件目标：**单卡 RTX 4060（8 GB 显存）**。

---

## 2. 目录结构

```text
poem_generation/
├── configs/default.yaml           # 全局配置（路径、超参、API、生成参数）
├── data/
│   ├── raw/                       # （软链/复制）chinese-poetry 原始数据
│   ├── preprocess.py              # 繁简转换 + 按格式分桶 + 清洗去重
│   └── processed/                 # 输出：tang_jueju5.txt 等
├── prompts/
│   └── test_prompts.json          # 50 条固定评测 prompt
├── models/
│   ├── char_lstm/                 # M1
│   │   ├── model.py / train.py / generate.py
│   ├── api_client/                # M2
│   │   ├── prompt_templates.py
│   │   └── run_api.py
│   └── gpt2_lora/                 # M3（可选）
│       ├── train_lora.py / generate.py
├── eval/
│   ├── auto_metrics.py            # 格式合规率 / 押韵率 / BLEU / ROUGE / 重复率
│   ├── human_eval_template.csv
│   └── results/                   # 各方法生成结果 + 指标 JSON
├── viz/
│   ├── plot_loss.py
│   ├── make_wordcloud.py
│   └── plot_metrics.py            # 柱状图 + 雷达图
├── app/
│   └── gradio_demo.py             # 统一交互界面（验收用）
├── scripts/                       # 一键脚本：preprocess / train_lstm / run_api / eval / plot
├── report/                        # 报告草稿
├── requirements.txt
└── README.md
```

---

## 3. 环境配置

> 推荐 Python 3.10+；CUDA 12.x（对应 4060）。

```powershell
# 1) 创建并激活虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) 安装 PyTorch（按 CUDA 版本选择，下例为 CUDA 12.1）
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3) 安装其余依赖
pip install -r requirements.txt
```

API Key（按需设置环境变量）：

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
$env:ZHIPU_API_KEY    = "..."
$env:DASHSCOPE_API_KEY= "..."
```

---

## 4. 一键流水线

```powershell
# 1. 数据预处理（繁→简、按格式分桶）
python data/preprocess.py --config configs/default.yaml

# 2. 训练 M1 Char-LSTM
python models/char_lstm/train.py --config configs/default.yaml

# 3. 用 M1 跑测试 prompt
python models/char_lstm/generate.py --config configs/default.yaml \
    --prompts prompts/test_prompts.json \
    --output eval/results/lstm.json

# 4. 调用 API（M2）
python models/api_client/run_api.py --config configs/default.yaml \
    --prompts prompts/test_prompts.json \
    --output_dir eval/results

# 5. （可选）M3 LoRA 微调 + 推理
python models/gpt2_lora/train_lora.py --config configs/default.yaml
python models/gpt2_lora/generate.py --config configs/default.yaml \
    --prompts prompts/test_prompts.json \
    --output eval/results/gpt2_lora.json

# 6. 自动评估
python eval/auto_metrics.py --results_dir eval/results --output eval/results/summary.csv

# 7. 可视化
python viz/plot_metrics.py --summary eval/results/summary.csv --out viz/

# 8. 启动演示界面
python app/gradio_demo.py --config configs/default.yaml
```

---

## 5. 数据与开源资源声明

| 资源 | 来源 | 用途 | 修改 |
|---|---|---|---|
| chinese-poetry | https://github.com/chinese-poetry/chinese-poetry (MIT) | 训练 + 评测语料 | 繁→简 / 按格式分桶 / 清洗去重 |
| OpenCC | https://github.com/BYVoid/OpenCC | 繁简转换 | 直接调用 |
| transformers / peft | HuggingFace | GPT-2 微调框架 | 直接调用 |
| DeepSeek / 通义 / 智谱 API | 官方 | M2 baseline | API 调用 |
| uer/gpt2-distil-chinese-cluecorpussmall | HuggingFace | M3 基座 | LoRA 微调 |

本组完成的工作（详见报告"本人分工内容"）：

1. 数据预处理脚本（繁简转换、格式判定、分桶）
2. Char-LSTM 模型 / 训练 / 生成（含藏头模式）的从零实现
3. Prompt 工程：3 种策略模板设计
4. GPT-2 LoRA 微调脚本
5. 评估指标：格式合规率、押韵率、重复率、BLEU/ROUGE
6. 可视化：训练曲线、词云、指标柱状图、人工评价雷达图
7. Gradio 演示界面

---

## 6. 评估指标与目标

| 指标 | 含义 | 目标 |
|---|---|---|
| 格式合规率 | 句数 + 每句字数是否符合所声明格式 | ≥ 80%（M2/M3）/ ≥ 50%（M1）|
| 押韵率 | 末字普通话韵母一致比例 | ≥ 60% |
| 重复率 (1-distinct) | n-gram 重复比例 | 越低越好 |
| BLEU-2/4 vs ROUGE-L | 与训练集真实诗的 n-gram 相似度 | 用于横向对比 |
| 人工评价 | 5 维度 5 分制 | 每方法 ≥ 10 条评分 |

---

## 7. 注意事项

- 验收时统一用 `prompts/test_prompts.json`，**不要边做边改**。
- 若 4060 训 LoRA 显存吃紧：将 `gpt2.batch_size` 降到 4 / 启 `gpt2.fp16: true`，或换用 0.5B 量级蒸馏模型。
- API 费用：DeepSeek 50 prompt × 6 组约 1 元；智谱 GLM-4-Flash 当前免费。
