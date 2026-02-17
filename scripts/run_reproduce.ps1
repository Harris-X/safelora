Param(
    [Parameter(Mandatory = $true)]
    [string]$BaseModelPath,

    [Parameter(Mandatory = $true)]
    [string]$ChatModelPath,

    [string]$TrainFile = "datasets/samsum_1000_bad.jsonl",
    [string]$TestFile = "datasets/samsum_test.jsonl",
    [string]$OutRoot = "outputs/reproduce",
    [int]$MaxEvalSamples = 200,
    [switch]$UseFp16
)

$ErrorActionPreference = "Stop"

Write-Host "[1/4] Installing dependencies..."
python -m pip install -r requirements-reproduce.txt

$LoraOut = Join-Path $OutRoot "lora_samsum_bad"
$SafeOut = Join-Path $OutRoot "lora_samsum_bad_safelora"

Write-Host "[2/4] LoRA fine-tuning on $TrainFile ..."
$trainArgs = @(
    "scripts/train_samsum_lora.py",
    "--chat_model_path", $ChatModelPath,
    "--train_file", $TrainFile,
    "--output_dir", $LoraOut,
    "--num_train_epochs", "5",
    "--learning_rate", "5e-5",
    "--lora_r", "8",
    "--target_modules", "q_proj,v_proj"
)
if ($UseFp16) { $trainArgs += "--fp16" }
python @trainArgs

Write-Host "[3/4] Applying SafeLoRA projection ..."
$safeArgs = @(
    "scripts/apply_safelora.py",
    "--chat_model_path", $ChatModelPath,
    "--base_model_path", $BaseModelPath,
    "--adapter_path", $LoraOut,
    "--output_adapter_path", $SafeOut,
    "--select_layers_type", "number",
    "--num_proj_layers", "7",
    "--threshold", "0.35"
)
if ($UseFp16) { $safeArgs += "--fp16" }
python @safeArgs

Write-Host "[4/4] Evaluating projected adapter on $TestFile ..."
$evalArgs = @(
    "scripts/eval_samsum_rouge.py",
    "--chat_model_path", $ChatModelPath,
    "--adapter_path", $SafeOut,
    "--test_file", $TestFile,
    "--max_samples", "$MaxEvalSamples"
)
if ($UseFp16) { $evalArgs += "--fp16" }
python @evalArgs

Write-Host "Done. Output adapters are under: $OutRoot"
