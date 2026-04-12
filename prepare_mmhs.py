
import json
import os
import argparse
from typing import List, Dict
import pandas as pd

def main():
	parser = argparse.ArgumentParser(description="Convert MMHS-150K to JSONL format for MMBT training.")
	parser.add_argument("--root", type=str, default="datasets/MMHS-150K", help="Root directory of MMHS-150K")
	args = parser.parse_args()

	root = args.root
	gt_path = os.path.join(root, "MMHS150K_GT.json")
	split_dir = os.path.join(root, "splits")
	img_dir = os.path.join(root, "img_resized")
	out_dir = os.path.join(root, "data")
	
	os.makedirs(out_dir, exist_ok=True)

	print(f"Loading GT from {gt_path}...")
	with open(gt_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	# 0: NotHate, 1+: Hate
	def get_binary_label(labels: List[int]) -> int:
		# Majority vote
		hate_count = sum(1 for x in labels if x > 0)
		not_hate_count = sum(1 for x in labels if x == 0)
		return 1 if hate_count > not_hate_count else 0

	splits = ["train", "val", "test"]
	# Map 'val' to 'dev' for consistency with codebase
	split_map = {"train": "train", "val": "dev", "test": "test"}

	for split in splits:
		ids_file = os.path.join(split_dir, f"{split}_ids.txt")
		if not os.path.exists(ids_file):
			print(f"Warning: Split file {ids_file} not found. Skipping.")
			continue
		
		print(f"Processing {split} split...")
		with open(ids_file, "r") as f:
			ids = [line.strip() for line in f if line.strip()]

		records = []
		for item_id in ids:
			if item_id not in data:
				continue
			
			item = data[item_id]
			# Ensure image exists
			img_name = f"{item_id}.jpg"
			if not os.path.exists(os.path.join(img_dir, img_name)):
				# Try png?
				if os.path.exists(os.path.join(img_dir, f"{item_id}.png")):
					img_name = f"{item_id}.png"
				else:
					# Skip missing images
					continue
			
			label = get_binary_label(item["labels"])
			text = item["tweet_text"]
			
			records.append({
				"img": img_name,
				"text": text,
				"label": label
			})
		
		out_split = split_map.get(split, split)
		out_file = os.path.join(out_dir, f"{out_split}.jsonl")
		
		# Save as JSONL
		df = pd.DataFrame(records)
		df.to_json(out_file, orient="records", lines=True)
		print(f"Saved {len(records)} records to {out_file}")

if __name__ == "__main__":
	main()
