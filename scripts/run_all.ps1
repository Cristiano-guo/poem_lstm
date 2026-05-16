# PowerShell one-click pipeline (Windows)
# Usage:
#   cd e:\calc-intel\lab\poem_generation
#   .\scripts\run_all.ps1               # run full pipeline
#   .\scripts\run_all.ps1 -SkipApi      # skip API calls
#   .\scripts\run_all.ps1 -SkipGpt2     # skip GPT-2 fine-tuning

param(
    [switch]$SkipApi   = $false,
    [switch]$SkipGpt2  = $true,    # skip GPT-2 by default (slow)
    [switch]$SkipLstm  = $false,
    [string]$Config    = "configs/default.yaml"
)

$ErrorActionPreference = "Stop"

function Step($msg) {
    Write-Host ("`n=== {0} ===" -f $msg) -ForegroundColor Cyan
}

Step "1) Data preprocessing"
python data/preprocess.py --config $Config

if (-not $SkipLstm) {
    Step "2) Char-LSTM training"
    python models/char_lstm/train.py --config $Config

    Step "3) Char-LSTM inference"
    python models/char_lstm/generate.py --config $Config `
        --prompts prompts/test_prompts.json `
        --output eval/results/lstm.json
} else {
    Write-Host "Skip LSTM" -ForegroundColor Yellow
}

if (-not $SkipApi) {
    Step "4) API inference"
    python models/api_client/run_api.py --config $Config `
        --prompts prompts/test_prompts.json `
        --output_dir eval/results
} else {
    Write-Host "Skip API" -ForegroundColor Yellow
}

if (-not $SkipGpt2) {
    Step "5) GPT-2-LoRA fine-tuning"
    python models/gpt2_lora/train_lora.py --config $Config

    Step "6) GPT-2-LoRA inference"
    python models/gpt2_lora/generate.py --config $Config `
        --prompts prompts/test_prompts.json `
        --output eval/results/gpt2_lora.json
} else {
    Write-Host "Skip GPT-2-LoRA" -ForegroundColor Yellow
}

Step "7) Auto evaluation"
python eval/auto_metrics.py --results_dir eval/results

Step "8) Visualization"
python viz/plot_loss.py --history ckpts/char_lstm/history.json --out viz/loss_curve.png
python viz/make_wordcloud.py --inputs data/processed/tang_jueju5.txt data/processed/tang_jueju7.txt --out viz/wordcloud.png
python viz/plot_metrics.py --summary eval/results/summary.csv

Step "9) Human eval CSV"
python eval/make_human_eval_csv.py --results_dir eval/results --out_csv eval/results/human_eval.csv

Write-Host "`nAll steps completed!" -ForegroundColor Green
Write-Host "Next: fill eval/results/human_eval.csv, then run viz/plot_human_radar.py" -ForegroundColor Green
Write-Host "Launch demo: python app/gradio_demo.py --config $Config" -ForegroundColor Green
