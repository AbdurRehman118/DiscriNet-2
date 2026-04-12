import os
import time
# Speed up Transformers import on Windows by avoiding TF/Flax probing and tokenizer multiprocessing noise
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import os
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, confusion_matrix
from transformers import CLIPProcessor, CLIPModel, get_cosine_schedule_with_warmup
from torch import amp as torch_amp
from typing import Iterable

from data import HatefulMemes
from model import MMBTCLIP


class FocalLoss(torch.nn.Module):
	def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
		super().__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.reduction = reduction

	def forward(self, inputs, targets):
		bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
		pt = torch.exp(-bce_loss)  # prevents nans when probability 0
		focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
		
		if self.reduction == "mean":
			return focal_loss.mean()
		elif self.reduction == "sum":
			return focal_loss.sum()
		return focal_loss



def choose_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
	candidates = np.linspace(0.1, 0.9, 81)
	best_f1 = -1.0
	best_thr = 0.5
	for t in candidates:
		y_hat = (y_prob >= t).astype(int)
		f1 = f1_score(y_true, y_hat)
		if f1 > best_f1:
			best_f1 = f1
			best_thr = float(t)
	return best_thr


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = None) -> dict:
	auc = roc_auc_score(y_true, y_prob)
	ap = average_precision_score(y_true, y_prob)
	if threshold is None:
		threshold = choose_threshold_by_f1(y_true, y_prob)
	y_hat = (y_prob >= threshold).astype(int)
	f1 = f1_score(y_true, y_hat)
	acc = accuracy_score(y_true, y_hat)
	return {"auc": auc, "ap": ap, "f1": f1, "acc": acc, "thr": threshold}


def build_dataloaders(args):
	try:
		processor = CLIPProcessor.from_pretrained(args.clip, use_fast=True)
	except Exception:
		# Fallback if fast processor not available
		processor = CLIPProcessor.from_pretrained(args.clip)
	train_set = HatefulMemes(args.img_dir, args.train_json, processor, split="train", aug=True)
	dev_set = HatefulMemes(args.img_dir, args.dev_json, processor, split="dev", aug=False)
	test_set = HatefulMemes(args.img_dir, args.test_json, processor, split="test", aug=False)

	train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
	dev_loader = DataLoader(dev_set, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
	test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
	return processor, train_loader, dev_loader, test_loader


def evaluate(model, dl, device, fp16: bool):
	model.eval()
	ys, ps = [], []
	with torch.no_grad():
		for batch in dl:
			labels = batch.pop("labels").numpy()
			batch.pop("meta", None)
			batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
			with torch_amp.autocast(device_type="cuda", enabled=(fp16 and device == "cuda")):
				logits = model(**batch)
				probs = torch.sigmoid(logits).detach().cpu().numpy()
			ys.append(labels)
			ps.append(probs)
	y = np.concatenate(ys)
	p = np.concatenate(ps)
	return y, p


def predict_with_meta(model, dl, device, fp16: bool):
	model.eval()
	metas, probs = [], []
	with torch.no_grad():
		for batch in dl:
			meta_list = batch.get("meta")
			batch.pop("labels", None)
			batch.pop("meta", None)
			batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
			with torch_amp.autocast(device_type="cuda", enabled=(fp16 and device == "cuda")):
				logits = model(**batch)
				p = torch.sigmoid(logits).detach().cpu().numpy()
			# meta may be a dict of lists (from default_collate) or a list of dicts
			if isinstance(meta_list, dict):
				length = len(next(iter(meta_list.values())))
				for i in range(length):
					metas.append({k: meta_list[k][i] for k in meta_list})
			else:
				metas.extend(meta_list)
			probs.extend(p.tolist())
	return metas, np.array(probs, dtype=np.float32)


def visualize_samples(args, model, processor, device, fp16: bool, out_dir: str, num_samples: int = 12):
	try:
		from PIL import Image
		import matplotlib.pyplot as plt
	except Exception as e:
		print("Visualization dependencies missing, skipping:", e)
		return

	model.eval()
	os.makedirs(out_dir, exist_ok=True)
	ds = HatefulMemes(args.img_dir, args.test_json, processor, split="test", aug=False)
	N = min(num_samples, len(ds))
	indices = random.sample(range(len(ds)), N)
	with torch.no_grad():
		for idx_i, i in enumerate(indices):
			item = ds[i]
			meta = item["meta"]
			labels = item.pop("labels")
			item = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in item.items() if k != "meta"}
			with torch_amp.autocast(device_type="cuda", enabled=(fp16 and device == "cuda")):
				logit = model(**item)
				prob = float(torch.sigmoid(logit).item())
			img_path = os.path.join(args.img_dir, meta["img_path"])
			img = Image.open(img_path).convert("RGB")
			fig = plt.figure(figsize=(4, 4))
			plt.imshow(img)
			plt.axis("off")
			plt.title(f"GT={meta['label']}  Pred={prob:.2f}\nText: {meta['text'][:70]}")
			fig.savefig(os.path.join(out_dir, f"sample_{idx_i}.png"), bbox_inches="tight")
			plt.close(fig)


def main():
	ap = argparse.ArgumentParser()
	# Data
	ap.add_argument("--img_dir", type=str, required=True)
	ap.add_argument("--train_json", type=str, required=True)
	ap.add_argument("--dev_json", type=str, required=True)
	ap.add_argument("--test_json", type=str, required=True)
	# Model
	ap.add_argument("--clip", type=str, default="openai/clip-vit-large-patch14", help="Stronger default backbone")
	ap.add_argument("--hdim", type=int, default=256)
	ap.add_argument("--lora", action="store_true")
	# Train
	ap.add_argument("--batch", type=int, default=16)
	ap.add_argument("--accum", type=int, default=2)
	ap.add_argument("--epochs", type=int, default=15)
	ap.add_argument("--lr", type=float, default=2e-4)
	ap.add_argument("--lr_backbone", type=float, default=None, help="Learning rate for CLIP backbone")
	ap.add_argument("--weight_decay", type=float, default=0.01)
	ap.add_argument("--pos_weight", type=float, default=1.0)
	ap.add_argument("--fp16", action="store_true")
	ap.add_argument("--num_workers", type=int, default=0)  # safer default on Windows
	# Schedule
	ap.add_argument("--warmup_ratio", type=float, default=0.05)
	# Unfreeze schedule
	ap.add_argument("--unfreeze_epoch", type=int, default=6)
	ap.add_argument("--unfreeze_clip", action="store_true")
	# I/O
	ap.add_argument("--out", type=str, default="runs/mmbt_clip")
	ap.add_argument("--resume", type=str, default="")

	args = ap.parse_args()

	device = "cuda" if torch.cuda.is_available() else "cpu"
	torch.backends.cudnn.benchmark = True

	os.makedirs(args.out, exist_ok=True)
	with open(os.path.join(args.out, "args.json"), "w") as f:
		json.dump(vars(args), f, indent=2)

	# Data
	processor, train_loader, dev_loader, test_loader = build_dataloaders(args)

	# Model
	# Always load in float32; autocast will handle mixed precision during forward
	# Loading in float16 directly causes GradScaler issues with optimizer
	dtype = torch.float32 
	clip_model = CLIPModel.from_pretrained(args.clip, torch_dtype=dtype)
	model = MMBTCLIP(clip_model, proj_dim=args.hdim, use_lora=args.lora).to(device)

	# Start with everything frozen inside CLIP; head trainable
	model.configure_trainable(enable_lora=False, unfreeze_clip=False)

	# Optimizer & scheduler
	def trainable_params(m, lr_backbone, lr_head) -> Iterable[dict]:
		# Separate backbone (CLIP) and head parameters
		backbone_params = []
		head_params = []
		
		for name, p in m.named_parameters():
			if not p.requires_grad:
				continue
			if "clip_model" in name:
				backbone_params.append(p)
			else:
				head_params.append(p)
		
		return [
			{"params": backbone_params, "lr": lr_backbone},
			{"params": head_params, "lr": lr_head}
		]

	# Use separate LRs if provided, else standard
	lr_backbone = args.lr_backbone if args.lr_backbone is not None else args.lr
	lr_head = args.lr
	
	optimizer = torch.optim.AdamW(trainable_params(model, lr_backbone, lr_head), weight_decay=args.weight_decay)
	total_steps = len(train_loader) * max(1, args.epochs)
	warmup_steps = max(1, int(args.warmup_ratio * total_steps))
	scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
	scaler = torch_amp.GradScaler("cuda") if (args.fp16 and device == "cuda") else None

	best_auc = -1.0
	best_path = None
	start_epoch = 1
	best_thr = 0.5

	# Resume if provided
	if args.resume and os.path.isfile(args.resume):
		ckpt = torch.load(args.resume, map_location=device)
		model.load_state_dict(ckpt["model"], strict=False)
		if "opt" in ckpt and "sched" in ckpt:
			try:
				# Optimizer state might mismatch if we changed param groups, so be careful
				# If structure changed, better to reset optimizer or handle gracefully
				optimizer.load_state_dict(ckpt["opt"])
				scheduler.load_state_dict(ckpt["sched"])
			except Exception as e:
				print(f"Warning: Could not load optimizer/scheduler state: {e}. restarting optimization.")
		start_epoch = int(ckpt.get("epoch", 0)) + 1
		best_auc = float(ckpt.get("auc", -1.0))
		best_thr = float(ckpt.get("thr", 0.5))
		print(f"Resumed from {args.resume} @ epoch {start_epoch} (best_auc={best_auc:.4f})")

	pos_weight = torch.tensor([args.pos_weight], device=device)
	# Use Focal Loss with pos_weight support implicitly via alpha or explicit BCE weight
	# Updating criterion to support pos_weight in BCE calculation if desired, 
	# but FocalLoss alpha already acts as class balance. 
	# Let's trust alpha=0.75 for now as "pos_weight" equivalent 
	# OR we can pass pos_weight to BCE if we modify FocalLoss.
	# User plan asked for "Weighted Loss: Add class weighting support".
	# Let's modify FocalLoss to take pos_weight.
	class WeightedFocalLoss(FocalLoss):
		def __init__(self, alpha=0.25, gamma=2.0, reduction="mean", pos_weight=None):
			super().__init__(alpha, gamma, reduction)
			self.pos_weight = pos_weight

		def forward(self, inputs, targets):
			# Pass pos_weight to BCE
			bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
				inputs, targets, reduction="none", pos_weight=self.pos_weight
			)
			pt = torch.exp(-bce_loss)
			focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
			if self.reduction == "mean": return focal_loss.mean()
			return focal_loss.sum()

	criterion = WeightedFocalLoss(alpha=0.75, gamma=2.0, pos_weight=pos_weight).to(device)


	# Training loop
	global_step = 0
	for epoch in range(start_epoch, args.epochs + 1):
		model.train()
		start_time = time.time()

		# Enable LoRA (and optionally base CLIP) after warmup epoch
		if epoch == args.unfreeze_epoch:
			print(f"Unfreezing components (CLIP={args.unfreeze_clip})...")
			model.configure_trainable(enable_lora=True, unfreeze_clip=args.unfreeze_clip)
			# Rebuild optimizer to include newly trainable params with correct LRs
			# N.B. Re-creating optimizer resets momentum. ideally we'd add param groups but this is simpler.
			optimizer = torch.optim.AdamW(trainable_params(model, lr_backbone, lr_head), weight_decay=args.weight_decay)
			# Re-create scheduler for remaining steps? Or continue?
			# If we reset optimizer, we should probably reset scheduler or adjust total_steps.
			# Simplified: just create new scheduler for remaining epochs.
			total_steps_remaining = len(train_loader) * (args.epochs - epoch + 1)
			warmup_steps_new = int(args.warmup_ratio * total_steps_remaining)
			scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps_new, num_training_steps=total_steps_remaining)

		running_loss = []
		optimizer.zero_grad(set_to_none=True)
		for step, batch in enumerate(train_loader, 1):
			labels = batch.pop("labels").to(device)
			batch.pop("meta", None)
			batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
			with torch_amp.autocast(device_type="cuda", enabled=(args.fp16 and device == "cuda")):
				logits = model(**batch)
				loss = criterion(logits, labels)
			if scaler is not None:
				scaler.scale(loss).backward()
			else:
				loss.backward()
			if step % args.accum == 0:
				if scaler is not None:
					scaler.unscale_(optimizer)
					torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
					scaler.step(optimizer)
					scaler.update()
				else:
					torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
					optimizer.step()
				optimizer.zero_grad(set_to_none=True)
				scheduler.step()
				global_step += 1
			
			if step % 50 == 0:
				elapsed = time.time() - start_time
				avg_time = elapsed / step
				remaining = (len(train_loader) - step) * avg_time
				# Format remaining as HH:MM:SS
				rem_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
				print(f"  Epoch {epoch} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f} | ETA: {rem_str}", end="\r")

			running_loss.append(loss.item())

		# Dev evaluation
		y_dev, p_dev = evaluate(model, dev_loader, device, args.fp16)
		dev_metrics = compute_metrics(y_dev, p_dev, threshold=None)
		print(f"[Epoch {epoch}] Loss={np.mean(running_loss):.4f} DevAUC={dev_metrics['auc']:.4f} F1={dev_metrics['f1']:.4f} Thr={dev_metrics['thr']:.2f}")

		# Save last & best
		last_path = os.path.join(args.out, "last.pt")
		torch.save(
			{
				"epoch": epoch,
				"model": model.state_dict(),
				"opt": optimizer.state_dict(),
				"sched": scheduler.state_dict(),
				"thr": dev_metrics["thr"],
				"auc": dev_metrics["auc"],
			},
			last_path,
		)
		if dev_metrics["auc"] > best_auc:
			best_auc = dev_metrics["auc"]
			best_thr = dev_metrics["thr"]
			best_path = os.path.join(args.out, "best.pt")
			torch.save({"epoch": epoch, "model": model.state_dict(), "thr": best_thr, "auc": best_auc}, best_path)
			print(f"  Saved BEST → {best_path}")

	# Final test using best (if available), else last
	load_path = best_path if best_path and os.path.isfile(best_path) else os.path.join(args.out, "last.pt")
	ckpt = torch.load(load_path, map_location=device)
	model.load_state_dict(ckpt["model"], strict=False)
	best_thr = float(ckpt.get("thr", best_thr))

	y_test, p_test = evaluate(model, test_loader, device, args.fp16)
	if np.min(y_test) < 0:
		# Unlabeled test set: skip metrics and dump predictions
		print("[TEST] Labels not found in test set. Saving predictions only.")
		metas, p_meta = predict_with_meta(model, test_loader, device, args.fp16)
		out_csv = os.path.join(args.out, "preds_test.csv")
		with open(out_csv, "w", encoding="utf-8") as f:
			f.write("img,text,prob\n")
			for m, prob in zip(metas, p_meta):
				text = m["text"].replace("\n", " ").replace(",", " ")
				f.write(f"{m['img_path']},{text},{float(prob):.6f}\n")
		print(f"Saved test predictions → {out_csv}")
	else:
		test_metrics = compute_metrics(y_test, p_test, threshold=best_thr)
		print(f"[TEST] AUC={test_metrics['auc']:.4f} AP={test_metrics['ap']:.4f} F1={test_metrics['f1']:.4f} Acc={test_metrics['acc']:.4f}")
		# Confusion matrix at best threshold
		y_hat = (p_test >= best_thr).astype(int)
		cm = confusion_matrix(y_test, y_hat)
		np.save(os.path.join(args.out, "confusion.npy"), cm)

	# Simple visualization
	visualize_samples(args, model, processor, device, args.fp16, args.out, num_samples=12)


if __name__ == "__main__":
	main()
