conda activate cv_cuda
Write-Host "Checking/Preparing MMHS-150K Dataset..."
python prepare_mmhs.py --root "datasets/MMHS-150K"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error preparing MMHS dataset. Please check if files are missing."
    Read-Host -Prompt "Press Enter to exit"
    exit
}

Write-Host "Starting Training on MMHS-150K Dataset..."

# Ensure output directory exists
if (!(Test-Path "runs/mmhs150k_vit_large")) {
    New-Item -ItemType Directory -Force -Path "runs/mmhs150k_vit_large"
}

python train.py `
  --img_dir "datasets/MMHS-150K/img_resized" `
  --train_json "datasets/MMHS-150K/data/train.jsonl" `
  --dev_json "datasets/MMHS-150K/data/dev.jsonl" `
  --test_json "datasets/MMHS-150K/data/test.jsonl" `
  --fp16 `
  --clip "openai/clip-vit-large-patch14" `
  --batch 4 `
  --accum 16 `
  --epochs 5 `
  --lr 2e-4 `
  --lr_backbone 1e-5 `
  --unfreeze_epoch 1 `
  --unfreeze_clip `
  --num_workers 0 `
  --out "runs/mmhs_retrain_v2"

# Note: Epochs set to 5 due to large dataset size (150k images)

Write-Host "Training Complete. Check 'runs/mmhs150k_vit_large' for results."
Read-Host -Prompt "Press Enter to exit"
