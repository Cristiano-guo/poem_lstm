# -*- coding: utf-8 -*-
"""三种 Prompt 策略：zero-shot / few-shot / CoT。

调用约定：build_prompt(strategy, prompt_dict) -> List[{'role','content'}]
"""
from __future__ import annotations

from typing import Dict, List

SYSTEM_BASIC = (
    '你是一位精通中国古典诗词的诗人，擅长按照五言、七言绝句和律诗、'
    '以及常见词牌格律进行创作。请严格遵守用户给出的格式要求与字数要求。'
    '只输出诗词正文，不要添加任何额外说明、注解或标题。'
)

SYSTEM_COT = (
    '你是一位精通中国古典诗词的诗人。请按以下步骤完成创作：\n'
    '1) 先在脑海中列出 3-5 个与主题相关的意象；\n'
    '2) 构思起承转合，使前后句意连贯；\n'
    '3) 检查字数、句数是否符合要求；\n'
    '4) 若是绝句或律诗，确保偶数句押韵（韵脚相同或相近）；\n'
    '5) 最终只输出诗词正文，每句用逗号或句号分隔，不要任何额外说明。'
)


# few-shot 示例库（极简，按格式类型）
FEW_SHOT_EXAMPLES: Dict[str, List[Dict[str, str]]] = {
    'jueju5': [
        {
            'instruction': '请以《春晓》为题，写一首五言绝句。',
            'answer': '春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。',
        },
        {
            'instruction': '请以《静夜思》为题，写一首五言绝句。',
            'answer': '床前明月光，疑是地上霜。举头望明月，低头思故乡。',
        },
    ],
    'jueju7': [
        {
            'instruction': '请以《早春》为题，写一首七言绝句。',
            'answer': '天街小雨润如酥，草色遥看近却无。最是一年春好处，绝胜烟柳满皇都。',
        },
        {
            'instruction': '请以《出塞》为题，写一首七言绝句。',
            'answer': '秦时明月汉时关，万里长征人未还。但使龙城飞将在，不教胡马度阴山。',
        },
    ],
    'lvshi5': [
        {
            'instruction': '请以《秋夜思乡》为题，写一首五言律诗。',
            'answer': (
                '戍鼓断人行，边秋一雁声。露从今夜白，月是故乡明。'
                '有弟皆分散，无家问死生。寄书长不达，况乃未休兵。'
            ),
        }
    ],
    'lvshi7': [
        {
            'instruction': '请以《登高》为题，写一首七言律诗。',
            'answer': (
                '风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。'
                '万里悲秋常作客，百年多病独登台。艰难苦恨繁霜鬓，潦倒新停浊酒杯。'
            ),
        }
    ],
    'ci': [
        {
            'instruction': '请按《如梦令》词牌格律创作一首词，主题为春日。',
            'answer': (
                '昨夜雨疏风骤，浓睡不消残酒。试问卷帘人，却道海棠依旧。'
                '知否？知否？应是绿肥红瘦。'
            ),
        }
    ],
}


def _format_user_query(p: dict) -> str:
    """把 prompt 字典渲染成自然语言指令。"""
    base = (p.get('instruction', '') or '').strip()
    fmt = p.get('format_spec', {}) or {}
    extra = []
    t = fmt.get('type')
    if t == 'jueju5':
        extra.append('【格式】五言绝句：四句，每句五字，偶数句押韵。')
    elif t == 'jueju7':
        extra.append('【格式】七言绝句：四句，每句七字，偶数句押韵。')
    elif t == 'lvshi5':
        extra.append('【格式】五言律诗：八句，每句五字，偶数句押韵，中间两联对仗。')
    elif t == 'lvshi7':
        extra.append('【格式】七言律诗：八句，每句七字，偶数句押韵，中间两联对仗。')
    elif t == 'ci' and fmt.get('rhythmic'):
        extra.append(f'【格式】词牌《{fmt["rhythmic"]}》，按词牌格律分句、押韵。')
    if fmt.get('acrostic'):
        extra.append(f'【藏头】每句首字依次为：{fmt["acrostic"]}。')
    if extra:
        return base + '\n' + '\n'.join(extra)
    return base


def build_messages(strategy: str, p: dict) -> List[Dict[str, str]]:
    user = _format_user_query(p)
    fmt_type = (p.get('format_spec', {}) or {}).get('type', 'jueju7')

    if strategy == 'zero_shot':
        return [
            {'role': 'system', 'content': SYSTEM_BASIC},
            {'role': 'user', 'content': user},
        ]

    if strategy == 'few_shot':
        msgs: List[Dict[str, str]] = [{'role': 'system', 'content': SYSTEM_BASIC}]
        examples = FEW_SHOT_EXAMPLES.get(fmt_type) or FEW_SHOT_EXAMPLES.get('jueju7', [])
        for ex in examples[:2]:
            msgs.append({'role': 'user', 'content': ex['instruction']})
            msgs.append({'role': 'assistant', 'content': ex['answer']})
        msgs.append({'role': 'user', 'content': user})
        return msgs

    if strategy == 'cot':
        return [
            {'role': 'system', 'content': SYSTEM_COT},
            {'role': 'user', 'content': user},
        ]

    raise ValueError(f'unknown strategy: {strategy}')
