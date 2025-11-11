import os
import json
from typing import Dict, Any

import torch
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset


class HatefulMemes(Dataset):
	"""
	Expects JSONL with columns:
	  - img: filename in img_dir (e.g., '01235.png')
	  - text: meme text/caption
	  - label: 0/1
	Images are loaded from img_dir and tokenized with a HuggingFace CLIPProcessor.
	"""

	def __init__(self, img_dir: str, jsonl_path: str, clip_processor, split: str = "train", aug: bool = False) -> None:
		self.img_dir = img_dir
		self.proc = clip_processor
		self.aug = aug
		# Read JSON Lines
		self.df = pd.read_json(jsonl_path, lines=True)
		# Normalize expected columns (allow some common variations)
		col_map = {}
		if "img" not in self.df.columns:
			for c in self.df.columns:
				if c.lower() in ("image", "image_name", "img_name", "filename", "file"):
					col_map[c] = "img"
					break
		if "text" not in self.df.columns:
			for c in self.df.columns:
				if c.lower() in ("caption", "sentence", "utterance"):
					col_map[c] = "text"
					break
		if "label" not in self.df.columns:
			for c in self.df.columns:
				if c.lower() in ("target", "y", "class"):
					col_map[c] = "label"
					break
		if col_map:
			self.df = self.df.rename(columns=col_map)

		# Require only img and text; label may be absent for test split
		missing = [c for c in ("img", "text") if c not in self.df.columns]
		if missing:
			raise ValueError(f"Missing required columns in {jsonl_path}: {missing}")

		# Normalize image paths to filenames (JSONL may contain 'img/XXXX.png')
		self.df["img"] = self.df["img"].astype(str).apply(lambda p: os.path.basename(p.replace("\\", "/")))

	def __len__(self) -> int:
		return len(self.df)

	def __getitem__(self, index: int) -> Dict[str, Any]:
		row = self.df.iloc[index]
		img_path = os.path.join(self.img_dir, row["img"])
		# Handle truncated/corrupt images gracefully
		ImageFile.LOAD_TRUNCATED_IMAGES = True
		try:
			image = Image.open(img_path).convert("RGB")
		except Exception as e:
			# Fallback to a blank image to avoid dataloader hangs
			print(f"[data] Warning: failed to load image '{img_path}': {e}. Using a blank image.")
			image = Image.new("RGB", (224, 224), (0, 0, 0))
		text = str(row["text"])
		# label may be missing (e.g., test set); use -1 sentinel when absent
		if "label" in self.df.columns and pd.notna(row["label"]):
			label_val = int(row["label"])
			has_label = True
		else:
			label_val = -1
			has_label = False
		label = torch.tensor(label_val, dtype=torch.float32)

		# Let CLIPProcessor handle resizing/normalization
		enc = self.proc(
			text=[text],
			images=image,
			return_tensors="pt",
			padding="max_length",
			truncation=True,
			max_length=77,  # CLIP text context length
		)
		enc = {k: v.squeeze(0) for k, v in enc.items()}  # remove batch dim
		enc["labels"] = label
		enc["meta"] = {"img_path": row["img"], "text": text, "label": label_val, "has_label": has_label}
		return enc


