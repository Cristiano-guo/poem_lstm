# -*- coding: utf-8 -*-
"""生成 prompts/test_prompts.json（避免被 Write 工具吞掉中文引号）。"""
import json
from pathlib import Path

LQ = '\u201c'  # “
RQ = '\u201d'  # ”

def q(s: str) -> str:
    """用 “”（U+201C/U+201D） 包裹字符串。"""
    return f'{LQ}{s}{RQ}'

themes = ['春雨', '边塞', '离别', '月夜', '江南']
input_types = [
    'topic_only', 'topic_keywords', 'first_line', 'acrostic', 'ci_pai',
    'image_description', 'style_imitation', 'format_jueju5', 'format_jueju7',
    'format_lvshi',
]

# 50 个 prompt
# 每个主题 10 条
prompts = []

def add(pid, theme, it, inp, instr, fmt):
    prompts.append({
        'id': pid, 'theme': theme, 'input_type': it,
        'input': inp, 'instruction': instr, 'format_spec': fmt,
    })

# ---- 春雨 ----
add('P01', '春雨', 'topic_only', {'topic': '春雨'},
    f'请以{q("春雨")}为主题，写一首七言绝句。要求每句七字，四句，偶数句押韵。',
    {'type': 'jueju7', 'lines': 4, 'chars_per_line': 7})
add('P02', '春雨', 'topic_keywords', {'topic': '春雨', 'keywords': ['细雨', '桃花', '燕子']},
    f'请围绕主题{q("春雨")}，并使用关键词{q("细雨、桃花、燕子")}，创作一首五言绝句。',
    {'type': 'jueju5', 'lines': 4, 'chars_per_line': 5})
add('P03', '春雨', 'first_line', {'first_line': '好雨知时节'},
    f'请以{q("好雨知时节")}为首句，续写完成一首五言绝句。',
    {'type': 'jueju5', 'lines': 4, 'chars_per_line': 5})
add('P04', '春雨', 'acrostic', {'acrostic_chars': '春雨绵绵'},
    '请写一首藏头诗（七言绝句），每句首字依次为：春、雨、绵、绵。',
    {'type': 'jueju7', 'lines': 4, 'chars_per_line': 7, 'acrostic': '春雨绵绵'})
add('P05', '春雨', 'ci_pai', {'rhythmic': '如梦令', 'topic': '春雨'},
    f'请按《如梦令》词牌格律创作一首词，主题为{q("春雨")}。',
    {'type': 'ci', 'rhythmic': '如梦令'})
add('P06', '春雨', 'image_description', {'image_caption': '江南小巷青石板路，细雨蒙蒙，撑伞女子缓行'},
    f'请根据画面描写{q("江南小巷青石板路，细雨蒙蒙，撑伞女子缓行")}，创作一首七言绝句。',
    {'type': 'jueju7', 'lines': 4, 'chars_per_line': 7})
add('P07', '春雨', 'style_imitation', {'topic': '春雨', 'style': '杜甫'},
    f'请模仿杜甫沉郁顿挫的风格，以{q("春雨")}为题写一首五言律诗。',
    {'type': 'lvshi5', 'lines': 8, 'chars_per_line': 5})
add('P08', '春雨', 'format_jueju5', {'topic': '春日喜雨'},
    f'请以{q("春日喜雨")}为题，写一首五言绝句。',
    {'type': 'jueju5', 'lines': 4, 'chars_per_line': 5})
add('P09', '春雨', 'format_jueju7', {'topic': '春雨夜行'},
    f'请以{q("春雨夜行")}为题，写一首七言绝句。',
    {'type': 'jueju7', 'lines': 4, 'chars_per_line': 7})
add('P10', '春雨', 'format_lvshi', {'topic': '春雨润物'},
    f'请以{q("春雨润物")}为题，写一首七言律诗。',
    {'type': 'lvshi7', 'lines': 8, 'chars_per_line': 7})

# ---- 边塞 ----
add('P11', '边塞', 'topic_only', {'topic': '边塞征戍'},
    f'请以{q("边塞征戍")}为主题，写一首七言绝句。',
    {'type': 'jueju7', 'lines': 4, 'chars_per_line': 7})
add('P12', '边塞', 'topic_keywords', {'topic': '边塞', 'keywords': ['大漠', '孤烟', '长河']},
    f'请以{q("边塞")}为主题，使用关键词{q("大漠、孤烟、长河")}，写一首五言绝句。',
    {'type': 'jueju5', 'lines': 4, 'chars_per_line': 5})
add('P13', '边塞', 'first_line', {'first_line': '葡萄美酒夜光杯'},
    f'请以{q("葡萄美酒夜光杯")}为首句，续写完成一首七言绝句。',
    {'type': 'jueju7', 'lines': 4, 'chars_per_line': 7})
add('P14', '边塞', 'acrostic', {'acrostic_chars': '塞外征人'},
    '请写一首藏头诗（五言绝句），每句首字依次为：塞、外、征、人。',
    {'type': 'jueju5', 'lines': 4, 'chars_per_line': 5, 'acrostic': '塞外征人'})
add('P15', '边塞', 'ci_pai', {'rhythmic': '渔家傲', 'topic': '边塞秋思'},
    f'请按《渔家傲》词牌格律创作一首词，主题为{q("边塞秋思")}。',
    {'type': 'ci', 'rhythmic': '渔家傲'})
add('P16', '边塞', 'image_description', {'image_caption': '黄沙漫天，烽火台上孤烟直上，远处铁骑列阵'},
    f'请根据画面{q("黄沙漫天，烽火台上孤烟直上，远处铁骑列阵")}，创作一首七言绝句。',
    {'type': 'jueju7', 'lines': 4, 'chars_per_line': 7})
add('P17', '边塞', 'style_imitation', {'topic': '边塞', 'style': '王昌龄'},
    f'请模仿王昌龄边塞诗的雄浑风格，写一首七言绝句，主题为{q("出塞")}。',
    {'type': 'jueju7', 'lines': 4, 'chars_per_line': 7})
add('P18', '边塞', 'format_jueju5', {'topic': '塞上闻笛'},
    f'请以{q("塞上闻笛")}为题，写一首五言绝句。',
    {'type': 'jueju5', 'lines': 4, 'chars_per_line': 5})
add('P19', '边塞', 'format_jueju7', {'topic': '凉州夜歌'},
    f'请以{q("凉州夜歌")}为题，写一首七言绝句。',
    {'type': 'jueju7', 'lines': 4, 'chars_per_line': 7})
add('P20', '边塞', 'format_lvshi', {'topic': '塞外征戍'},
    f'请以{q("塞外征戍")}为题，写一首五言律诗。',
    {'type': 'lvshi5', 'lines': 8, 'chars_per_line': 5})

# ---- 离别 ----
add('P21', '离别', 'topic_only', {'topic': '送别友人'},
    f'请以{q("送别友人")}为主题，写一首五言绝句。',
    {'type': 'jueju5', 'lines': 4, 'chars_per_line': 5})
add('P22', '离别', 'topic_keywords', {'topic': '离别', 'keywords': ['长亭', '杨柳', '孤帆']},
    f'请以{q("离别")}为主题，使用关键词{q("长亭、杨柳、孤帆")}，写一首七言绝句。',
    {'type': 'jueju7', 'lines': 4, 'chars_per_line': 7})
add('P23', '离别', 'first_line', {'first_line': '孤帆远影碧空尽'},
    f'请以{q("孤帆远影碧空尽")}为首句，续写完成一首七言绝句。',
    {'type': 'jueju7', 'lines': 4, 'chars_per_line': 7})
add('P24', '离别', 'acrostic', {'acrostic_chars': '送君千里'},
    '请写一首藏头诗（七言绝句），每句首字依次为：送、君、千、里。',
    {'type': 'jueju7', 'lines': 4, 'chars_per_line': 7, 'acrostic': '送君千里'})
add('P25', '离别', 'ci_pai', {'rhythmic': '雨霖铃', 'topic': '别离'},
    f'请按《雨霖铃》词牌格律创作一首词，主题为{q("别离")}。',
    {'type': 'ci', 'rhythmic': '雨霖铃'})
add('P26', '离别', 'image_description', {'image_caption': '渡口杨柳依依，友人挥手登舟，江上烟波茫茫'},
    f'请根据画面{q("渡口杨柳依依，友人挥手登舟，江上烟波茫茫")}，写一首七言绝句。',
    {'type': 'jueju7', 'lines': 4, 'chars_per_line': 7})
add('P27', '离别', 'style_imitation', {'topic': '送别', 'style': '王维'},
    f'请模仿王维清新自然的风格，写一首五言绝句，主题为{q("山中送别")}。',
    {'type': 'jueju5', 'lines': 4, 'chars_per_line': 5})
add('P28', '离别', 'format_jueju5', {'topic': '灞桥折柳'},
    f'请以{q("灞桥折柳")}为题，写一首五言绝句。',
    {'type': 'jueju5', 'lines': 4, 'chars_per_line': 5})
add('P29', '离别', 'format_jueju7', {'topic': '黄鹤楼送别'},
    f'请以{q("黄鹤楼送别")}为题，写一首七言绝句。',
    {'type': 'jueju7', 'lines': 4, 'chars_per_line': 7})
add('P30', '离别', 'format_lvshi', {'topic': '羁旅惜别'},
    f'请以{q("羁旅惜别")}为题，写一首七言律诗。',
    {'type': 'lvshi7', 'lines': 8, 'chars_per_line': 7})

# ---- 月夜 ----
add('P31', '月夜', 'topic_only', {'topic': '月夜思乡'},
    f'请以{q("月夜思乡")}为主题，写一首五言绝句。',
    {'type': 'jueju5', 'lines': 4, 'chars_per_line': 5})
add('P32', '月夜', 'topic_keywords', {'topic': '月夜', 'keywords': ['明月', '霜', '孤影']},
    f'请以{q("月夜")}为主题，使用关键词{q("明月、霜、孤影")}，写一首五言绝句。',
    {'type': 'jueju5', 'lines': 4, 'chars_per_line': 5})
add('P33', '月夜', 'first_line', {'first_line': '床前明月光'},
    f'请以{q("床前明月光")}为首句，续写完成一首五言绝句。',
    {'type': 'jueju5', 'lines': 4, 'chars_per_line': 5})
add('P34', '月夜', 'acrostic', {'acrostic_chars': '明月清风'},
    '请写一首藏头诗（五言绝句），每句首字依次为：明、月、清、风。',
    {'type': 'jueju5', 'lines': 4, 'chars_per_line': 5, 'acrostic': '明月清风'})
add('P35', '月夜', 'ci_pai', {'rhythmic': '水调歌头', 'topic': '中秋望月'},
    f'请按《水调歌头》词牌格律创作一首词，主题为{q("中秋望月")}。',
    {'type': 'ci', 'rhythmic': '水调歌头'})
add('P36', '月夜', 'image_description', {'image_caption': '深秋庭院，明月当空，落叶满阶，独坐听虫'},
    f'请根据画面{q("深秋庭院，明月当空，落叶满阶，独坐听虫")}，写一首七言绝句。',
    {'type': 'jueju7', 'lines': 4, 'chars_per_line': 7})
add('P37', '月夜', 'style_imitation', {'topic': '月夜独酌', 'style': '李白'},
    f'请模仿李白飘逸豪放的风格，写一首五言绝句，主题为{q("月夜独酌")}。',
    {'type': 'jueju5', 'lines': 4, 'chars_per_line': 5})
add('P38', '月夜', 'format_jueju5', {'topic': '山月独行'},
    f'请以{q("山月独行")}为题，写一首五言绝句。',
    {'type': 'jueju5', 'lines': 4, 'chars_per_line': 5})
add('P39', '月夜', 'format_jueju7', {'topic': '江楼望月'},
    f'请以{q("江楼望月")}为题，写一首七言绝句。',
    {'type': 'jueju7', 'lines': 4, 'chars_per_line': 7})
add('P40', '月夜', 'format_lvshi', {'topic': '秋夜思怀'},
    f'请以{q("秋夜思怀")}为题，写一首五言律诗。',
    {'type': 'lvshi5', 'lines': 8, 'chars_per_line': 5})

# ---- 江南 ----
add('P41', '江南', 'topic_only', {'topic': '江南春景'},
    f'请以{q("江南春景")}为主题，写一首七言绝句。',
    {'type': 'jueju7', 'lines': 4, 'chars_per_line': 7})
add('P42', '江南', 'topic_keywords', {'topic': '江南', 'keywords': ['杏花', '烟雨', '乌篷']},
    f'请以{q("江南")}为主题，使用关键词{q("杏花、烟雨、乌篷")}，写一首七言绝句。',
    {'type': 'jueju7', 'lines': 4, 'chars_per_line': 7})
add('P43', '江南', 'first_line', {'first_line': '日出江花红胜火'},
    f'请以{q("日出江花红胜火")}为首句，续写完成一首七言绝句。',
    {'type': 'jueju7', 'lines': 4, 'chars_per_line': 7})
add('P44', '江南', 'acrostic', {'acrostic_chars': '江南好景'},
    '请写一首藏头诗（七言绝句），每句首字依次为：江、南、好、景。',
    {'type': 'jueju7', 'lines': 4, 'chars_per_line': 7, 'acrostic': '江南好景'})
add('P45', '江南', 'ci_pai', {'rhythmic': '浣溪沙', 'topic': '江南春日'},
    f'请按《浣溪沙》词牌格律创作一首词，主题为{q("江南春日")}。',
    {'type': 'ci', 'rhythmic': '浣溪沙'})
add('P46', '江南', 'image_description', {'image_caption': '西湖断桥残雪，远山如黛，柳枝拂水'},
    f'请根据画面{q("西湖断桥残雪，远山如黛，柳枝拂水")}，写一首七言绝句。',
    {'type': 'jueju7', 'lines': 4, 'chars_per_line': 7})
add('P47', '江南', 'style_imitation', {'topic': '江南忆', 'style': '白居易'},
    f'请模仿白居易《忆江南》的风格，写一首小令，主题为{q("江南忆")}。',
    {'type': 'ci', 'rhythmic': '忆江南'})
add('P48', '江南', 'format_jueju5', {'topic': '江畔独步'},
    f'请以{q("江畔独步")}为题，写一首五言绝句。',
    {'type': 'jueju5', 'lines': 4, 'chars_per_line': 5})
add('P49', '江南', 'format_jueju7', {'topic': '苏堤春晓'},
    f'请以{q("苏堤春晓")}为题，写一首七言绝句。',
    {'type': 'jueju7', 'lines': 4, 'chars_per_line': 7})
add('P50', '江南', 'format_lvshi', {'topic': '秦淮夜泊'},
    f'请以{q("秦淮夜泊")}为题，写一首七言律诗。',
    {'type': 'lvshi7', 'lines': 8, 'chars_per_line': 7})

data = {
    'version': '1.0',
    'description': '吟诗作赋实验固定评测 prompt 集，共 50 条，按 5 大主题 × 10 类输入形式组织。所有方法使用同一份 prompt 评测。',
    'themes': themes,
    'input_types': input_types,
    'prompts': prompts,
}

out = Path(__file__).resolve().parents[1] / 'prompts' / 'test_prompts.json'
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f'[ok] {len(prompts)} prompts -> {out}')
