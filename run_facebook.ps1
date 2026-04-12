conda activate cv_cuda
Write-Host "Starting Training on Facebook Memes Dataset..."

# Ensure output directory exists to avoid errors
if (!(Test-Path "runs/fb_memes_vit_large")) {
    New-Item -ItemType Directory -Force -Path "runs/fb_memes_vit_large"
}

# Auto-resume logic
$resumeArg = ""
if (Test-Path "runs/fb_memes_vit_large/last.pt") {
    Write-Host "Resuming from checkpoint 'runs/fb_memes_vit_large/last.pt'..."
    $resumeArg = "--resume runs/fb_memes_vit_large/last.pt"
}

if ($resumeArg) {
    python train.py `
      --img_dir "datasets/Facebook Memes/data/img" `
      --train_json "datasets/Facebook Memes/data/train.jsonl" `
      --dev_json "datasets/Facebook Memes/data/dev.jsonl" `
      --test_json "datasets/Facebook Memes/data/test.jsonl" `
      --fp16 `
      --clip "openai/clip-vit-large-patch14" `
      --lora `
      --batch 16 `
      --accum 4 `
      --epochs 10 `
      --lr 2e-4 `
      --unfreeze_epoch 0 `
      --pos_weight 2.0 `
      --num_workers 0 `
      --out "runs/fb_memes_vit_large" `
      --resume "runs/fb_memes_vit_large/last.pt"
} else {
    python train.py `
      --img_dir "datasets/Facebook Memes/data/img" `
      --train_json "datasets/Facebook Memes/data/train.jsonl" `
      --dev_json "datasets/Facebook Memes/data/dev.jsonl" `
      --test_json "datasets/Facebook Memes/data/test.jsonl" `
      --fp16 `
      --clip "openai/clip-vit-large-patch14" `
      --lora `
      --batch 16 `
      --accum 4 `
      --epochs 10 `
      --lr 2e-4 `
      --unfreeze_epoch 0 `
      --pos_weight 2.0 `
      --num_workers 0 `
      --out "runs/fb_memes_vit_large"
}

Write-Host "Training Complete. Check 'runs/fb_memes_vit_large' for results."
