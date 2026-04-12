conda activate cv_cuda
Write-Host "Starting Combined Training (MMHS + FB Memes)..."

# Ensure output directory exists
if (!(Test-Path "runs/combined_mmbt_v1")) {
    New-Item -ItemType Directory -Force -Path "runs/combined_mmbt_v1"
}

# Check for resume
$resumeArg = ""
if (Test-Path "runs/combined_mmbt_v1/last.pt") {
    Write-Host "Resuming from checkpoint..."
    $resumeArg = "--resume runs/combined_mmbt_v1/last.pt"
}

# Note: passing "." as img_dir because JSONL has absolute paths now.
# Use $resumeArg
# invoke-expression or just pass it? better to construct command or use array splatting but simple string concat works for simple args if careful.
# best way in PS with backticks is to add it conditionally.

if ($resumeArg) {
    python train.py `
      --img_dir "." `
      --train_json "datasets/Combined/train.jsonl" `
      --dev_json "datasets/Combined/dev.jsonl" `
      --test_json "datasets/Combined/test.jsonl" `
      --fp16 `
      --clip "openai/clip-vit-large-patch14" `
      --lora `
      --batch 8 `
      --accum 8 `
      --epochs 3 `
      --lr 2e-4 `
      --unfreeze_epoch 1 `
      --pos_weight 2.0 `
      --num_workers 0 `
      --out "runs/combined_mmbt_v1" `
      --resume "runs/combined_mmbt_v1/last.pt"
} else {
    python train.py `
      --img_dir "." `
      --train_json "datasets/Combined/train.jsonl" `
      --dev_json "datasets/Combined/dev.jsonl" `
      --test_json "datasets/Combined/test.jsonl" `
      --fp16 `
      --clip "openai/clip-vit-large-patch14" `
      --lora `
      --batch 8 `
      --accum 8 `
      --epochs 3 `
      --lr 2e-4 `
      --unfreeze_epoch 1 `
      --pos_weight 2.0 `
      --num_workers 0 `
      --out "runs/combined_mmbt_v1"
}


Write-Host "Training Complete. Check 'runs/combined_mmbt_v1' for results."
