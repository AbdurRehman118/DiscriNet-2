import argparse
import json
import os
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


def load_policies(path: str) -> List[Dict[str, Any]]:
	policies: List[Dict[str, Any]] = []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			policies.append(json.loads(line))
	if not policies:
		raise ValueError(f"No policies found in {path}")
	return policies


def build_index(policies_path: str, clip_name: str, out_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
	policies = load_policies(policies_path)
	texts = [p.get("text", "") for p in policies]
	ids = [p.get("id", i) for i, p in enumerate(policies)]

	print(f"[policy_rag] Loaded {len(texts)} policies from {policies_path}")

	device = device or ("cuda" if torch.cuda.is_available() else "cpu")
	dtype = torch.float32
	processor = CLIPProcessor.from_pretrained(clip_name)
	model = CLIPModel.from_pretrained(clip_name, torch_dtype=dtype).to(device)
	model.eval()

	all_embs: List[torch.Tensor] = []
	batch_size = 16
	with torch.no_grad():
		for i in range(0, len(texts), batch_size):
			chunk = texts[i : i + batch_size]
			enc = processor(
				text=chunk,
				return_tensors="pt",
				padding=True,
				truncation=True,
				max_length=77,
			)
			enc = {k: v.to(device) for k, v in enc.items()}
			feats = model.get_text_features(**enc)
			feats = F.normalize(feats, p=2, dim=-1)
			all_embs.append(feats.cpu())

	embeddings = torch.cat(all_embs, dim=0)

	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	torch.save(
		{
			"clip": clip_name,
			"embeddings": embeddings,
			"texts": texts,
			"ids": ids,
		},
		out_path,
	)
	print(f"[policy_rag] Saved index with {len(texts)} policies → {out_path}")


def main() -> None:
	ap = argparse.ArgumentParser(description="Build CLIP-based policy RAG index.")
	ap.add_argument("mode", choices=["build"], help="Operation mode")
	ap.add_argument("--policies", type=str, required=True, help="Path to policies JSONL")
	ap.add_argument("--clip", type=str, default="openai/clip-vit-base-patch32", help="CLIP model name")
	ap.add_argument("--out", type=str, required=True, help="Output index path (.pt)")
	args = ap.parse_args()

	if args.mode == "build":
		build_index(args.policies, args.clip, args.out)


if __name__ == "__main__":
	main()


