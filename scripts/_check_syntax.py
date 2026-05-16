# -*- coding: utf-8 -*-
import ast
import json
import sys

py_files = [
    'data/preprocess.py',
    'models/char_lstm/model.py',
    'models/char_lstm/vocab.py',
    'models/char_lstm/dataset.py',
    'models/char_lstm/train.py',
    'models/char_lstm/generate.py',
    'models/api_client/providers.py',
    'models/api_client/prompt_templates.py',
    'models/api_client/run_api.py',
    'models/gpt2_lora/dataset.py',
    'models/gpt2_lora/train_lora.py',
    'models/gpt2_lora/generate.py',
    'eval/metrics.py',
    'eval/auto_metrics.py',
    'eval/make_human_eval_csv.py',
    'viz/plot_loss.py',
    'viz/make_wordcloud.py',
    'viz/plot_metrics.py',
    'viz/plot_human_radar.py',
    'app/gradio_demo.py',
    'scripts/check_env.py',
]

errs = 0
for f in py_files:
    try:
        ast.parse(open(f, encoding='utf-8').read(), f)
        print('ok', f)
    except SyntaxError as e:
        errs += 1
        print('ERR', f, e)

try:
    data = json.load(open('prompts/test_prompts.json', encoding='utf-8'))
    n = len(data.get('prompts', []))
    print(f'json ok prompts/test_prompts.json ({n} prompts)')
except Exception as e:
    errs += 1
    print('JSON ERR', e)

sys.exit(errs)
