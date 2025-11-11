# MMBT-CLIP Hateful Memes (RTX-3060 Friendly)

Lightweight multimodal (image + text) classifier for the Facebook Hateful Memes dataset.  
Uses CLIP encoders with a tiny bi-transformer fusion head, fp16, gradient accumulation, and optional LoRA adapters—designed to fit comfortably on a 12 GB RTX-3060.

---

## 1. Environment Setup

1. Install Python 3.10+ (Anaconda/Miniconda recommended).
2. Create/activate your environment (example with `conda`):

   ```powershell
   conda create -n hateful_memes python=3.10
   conda activate hateful_memes
   ```

3. Install dependencies:

   ```powershell
   pip install -r env.txt
   ```

---

## 2. Dataset Layout

Place the official Hateful Memes JSONL splits and images under `archive/data/`:

```
archive/data/
├── img/
│   ├── 01235.png
│   ├── ...
├── train.jsonl
├── dev.jsonl
└── test.jsonl   # labels may be missing (official test split)
```

- `img` paths in JSONL can include prefixes like `img/`; the loader strips them automatically.
- If your JSONL column names differ (e.g., `filename`), common variants are auto-mapped.

---

## 3. Training (from scratch)

Run from the repository root:

```powershell
python train.py `
  --img_dir "archive/data/img" `
  --train_json "archive/data/train.jsonl" `
  --dev_json "archive/data/dev.jsonl" `
  --test_json "archive/data/test.jsonl" `
  --fp16 `
  --lora `
  --unfreeze_epoch 6 `
  --batch 16 `
  --accum 2 `
  --epochs 15 `
  --out "runs/mmbt_clip_b32"
```

Key flags:
- `--fp16` enables mixed precision.
- `--lora` enables LoRA adapters (keeps VRAM low).
- `--unfreeze_epoch 6` starts LoRA/base fine-tuning after 6 epochs.
- `--batch 16 --accum 2` = effective batch size 32 with grad accumulation.

Output directory (`--out`) stores:
- `best.pt` (best dev ROC-AUC, includes threshold)
- `last.pt` (latest checkpoint with optimizer/scheduler)
- `args.json` (run configuration)
- `confusion.npy` (if test labels exist)
- `sample_*.png` (visualizations)
- `preds_test.csv` (test predictions if labels missing)

---

## 4. Resume Training

To resume from the latest checkpoint:

```powershell
python train.py `
  --img_dir "archive/data/img" `
  --train_json "archive/data/train.jsonl" `
  --dev_json "archive/data/dev.jsonl" `
  --test_json "archive/data/test.jsonl" `
  --fp16 `
  --lora `
  --unfreeze_epoch 6 `
  --batch 16 `
  --accum 2 `
  --epochs 15 `
  --out "runs/mmbt_clip_b32" `
  --resume "runs/mmbt_clip_b32/last.pt"
```

Make sure flags match the original run so optimizer shapes are consistent.

---

## 5. Evaluation & Outputs

During training:
- Primary metric: ROC-AUC (dev set).
- Secondary metrics: PR-AUC, F1, Accuracy at the dev-selected threshold.
- Best checkpoint, selected by dev AUC, stores the threshold (`thr`) for inference.

After training:
- If the test split has labels → prints test metrics and saves `confusion.npy`.
- If test labels are absent (official release) → saves `preds_test.csv` with columns `img,text,prob`.

To inspect the saved threshold:

```powershell
python - << 'PY'
import torch
ckpt = torch.load("runs/mmbt_clip_b32/best.pt", map_location="cpu")
print("Best dev AUC:", ckpt["auc"], "Threshold:", ckpt["thr"])
PY
```

---

## 6. Tips & Tweaks

- **Fast processor**: add `--use_fast_processor` flag (or edit `train.py`) to use CLIP’s fast image processor.
- **DataLoader workers**: on Windows keep `--num_workers 0` (default). Increase gradually if stable.
- **SDPA warning**: harmless cuDNN note about attention strides. To silence:

  ```python
  from torch.backends.cuda import sdp_kernel
  sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False)
  ```

- **Checkpoint safety**: when loading from untrusted sources, set `torch.load(..., weights_only=True)` (PyTorch >= 2.2).

---

## 7. FAQ

- **VRAM usage**: ~10–11 GB on RTX-3060 with the default config (fp16 + grad accumulation).
- **LoRA optional?** Yes. Omit `--lora` to train just the fusion head.
- **Change projection dim?** `--hdim 256` (default). Increase to 384 if memory allows.
- **Different splits/paths?** Point `--img_dir`, `--train_json`, etc., to your locations.

---

Happy training! For debugging or extension ideas (SigLIP, FiLM, contrastive loss), see comments in `train.py` / `model.py`.


