import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


class PolicyRAGScorer:
	def __init__(self, index_path: str, device: str | None = None) -> None:
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		data = torch.load(index_path, map_location="cpu")
		self.clip_name: str = data["clip"]
		self.policy_embs: torch.Tensor = data["embeddings"].to(self.device)
		self.policy_texts: List[str] = list(data["texts"])

		self.processor = CLIPProcessor.from_pretrained(self.clip_name)
		self.model = CLIPModel.from_pretrained(self.clip_name, torch_dtype=torch.float32).to(self.device)
		self.model.eval()

	def _encode_texts(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
		all_feats: List[torch.Tensor] = []
		with torch.no_grad():
			for i in range(0, len(texts), batch_size):
				chunk = texts[i : i + batch_size]
				enc = self.processor(
					text=chunk,
					return_tensors="pt",
					padding=True,
					truncation=True,
					max_length=77,
				)
				enc = {k: v.to(self.device) for k, v in enc.items()}
				feats = self.model.get_text_features(**enc)
				feats = F.normalize(feats, p=2, dim=-1)
				all_feats.append(feats)
		return torch.cat(all_feats, dim=0)

	def score(self, texts: List[str], calibration: str = "percentile") -> np.ndarray:
		"""
		Return a policy-consistency score in [0,1] for each text.
		Higher = more similar to at least one hate-speech policy description.
		
		Args:
			calibration: "percentile" (normalize by distribution) or "raw" (simple mapping)
		"""
		text_embs = self._encode_texts(texts)
		# cosine similarity to each policy embedding
		sims = text_embs @ self.policy_embs.T  # [N_text, N_policy]
		max_sim, _ = sims.max(dim=1)  # [N_text]
		max_sim = max_sim.clamp(-1.0, 1.0)
		
		if calibration == "raw":
			# Simple mapping: cosine [-1,1] -> [0,1]
			scores = (max_sim + 1.0) / 2.0
		elif calibration == "percentile":
			# Percentile-based calibration: map to match typical MMBT score distribution
			# Assume cosine similarities are in [0.5, 1.0] range typically
			# Map to [0, 1] with a more selective curve
			# Use sigmoid-like transformation with threshold
			# cos_sim > 0.85 -> high score, < 0.7 -> low score
			threshold = 0.75  # cosine threshold for "high risk"
			scale = 10.0  # steepness
			# Sigmoid: maps threshold to ~0.5, above threshold -> higher scores
			scores = torch.sigmoid(scale * (max_sim - threshold))
		elif calibration == "selective":
			# More conservative: only very high cosine similarities get high scores
			# cos > 0.9 -> 0.8-1.0, cos < 0.8 -> 0.0-0.3
			scores = torch.clamp((max_sim - 0.8) / 0.2, 0.0, 1.0)
		else:
			raise ValueError(f"Unknown calibration: {calibration}")
		
		return scores.detach().cpu().numpy()


def load_ckpt_threshold(ckpt_path: str) -> Tuple[float, float]:
	ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
	return float(ckpt.get("auc", 0.0)), float(ckpt.get("thr", 0.5))


def run(args: argparse.Namespace) -> None:
    # Load model predictions
    df = pd.read_csv(args.preds_csv)
    if "prob" not in df.columns:
        raise ValueError(f"{args.preds_csv} must contain a 'prob' column.")

    texts = df["text"].astype(str).tolist()
    p_model = df["prob"].astype(float).values

    # Policy RAG scores
    rag = PolicyRAGScorer(args.policy_index, device=None)
    p_policy = rag.score(texts, calibration=args.policy_calibration)

    # OLID Scores (Optional)
    p_olid = None
    if args.olid_csv:
        if not os.path.exists(args.olid_csv):
             raise FileNotFoundError(f"OLID CSV not found: {args.olid_csv}")
        olid_df = pd.read_csv(args.olid_csv)
        # Verify alignment or merge? Assuming same order for now, but check length
        if len(olid_df) != len(df):
             print(f"Warning: OLID CSV length ({len(olid_df)}) != Preds CSV length ({len(df)}). merging on 'text' if possible?")
             # For safety, let's assume strict alignment or fail.
             # But if they come from same source, they should match.
             pass
        
        # Look for 'olid_prob' or 'prob'
        if "olid_prob" in olid_df.columns:
             p_olid = olid_df["olid_prob"].astype(float).values
        elif "prob" in olid_df.columns:
             p_olid = olid_df["prob"].astype(float).values
        else:
             raise ValueError(f"OLID CSV must have 'olid_prob' or 'prob' column")

    # Ensemble
    alpha = args.alpha
    if p_olid is not None:
        # Distribution: alpha for model, rest split equally
        remain = 1.0 - alpha
        p_final = alpha * p_model + (remain / 2) * p_policy + (remain / 2) * p_olid
        print(f"[ensemble] Components: Model ({alpha}), Policy ({remain/2:.2f}), OLID ({remain/2:.2f})")
    else:
        p_final = alpha * p_model + (1.0 - alpha) * p_policy
        print(f"[ensemble] Components: Model ({alpha}), Policy ({1.0-alpha:.2f})")

    # Threshold from best checkpoint or custom
    auc, ckpt_thr = load_ckpt_threshold(args.ckpt)
    thr = args.threshold if args.threshold is not None else ckpt_thr
    print(f"[ensemble] Using checkpoint AUC={auc:.4f}, threshold={thr:.3f} {'(custom)' if args.threshold is not None else '(from checkpoint)'}")

    if args.max_voting:
        # Max-Voting Strategy (Logical OR)
        # If any model is confident > threshold, predict HATE
        # Note: We can use specific thresholds for output, but simple approach is global threshold
        print("[ensemble] Strategy: Max-Voting (Logical OR)")
        
        # Calculate individual decisions
        # Scale Policy to be comparable? It's already in [0,1].
        # MMBT
        l_model = (p_model >= thr).astype(int)
        
        # Policy RAG (Often has different distribution, maybe use a fixed 0.6?)
        # For now use same thr or 0.5 default for policy if thr is extreme
        policy_thr = thr if 0.3 <= thr <= 0.7 else 0.5
        l_policy = (p_policy >= policy_thr).astype(int)
        
        l_olid = np.zeros_like(l_model)
        if p_olid is not None:
            # OLID might need its own threshold (e.g. 0.5)
            # Assuming OLID prob is calibrated to [0,1]
            l_olid = (p_olid >= 0.5).astype(int) 
            
        # Logical OR
        label_final = np.maximum(l_model, np.maximum(l_policy, l_olid))
        
        # For p_final, we can just return the max probability as a proxy for confidence?
        # Or just keep the weighted average for ranking purposes?
        # Let's use max probability 
        p_final = np.maximum(p_model, p_policy)
        if p_olid is not None:
            p_final = np.maximum(p_final, p_olid)
            
    else:
        # Weighted Average Strategy
        label_final = (p_final >= thr).astype(int)

    # Save
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    
    out_data = {
        "img": df["img"],
        "text": df["text"],
        "prob_model": p_model,
        "policy_score": p_policy,
    }
    if p_olid is not None:
        out_data["prob_olid"] = p_olid
        
    out_data["prob_final"] = p_final
    out_data["label_final"] = label_final
    
    out_df = pd.DataFrame(out_data)
    out_df.to_csv(out_path, index=False)
    print(f"[ensemble] Saved ensemble predictions → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ensemble MMBT-CLIP predictions with policy RAG scoring.")
    ap.add_argument("--preds_csv", type=str, required=True, help="CSV with model predictions (img,text,prob)")
    ap.add_argument("--policy_index", type=str, required=True, help="Policy index built by policy_rag.py")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to best.pt checkpoint (for threshold)")
    ap.add_argument("--alpha", type=float, default=0.7, help="Weight for model prob in ensemble")
    ap.add_argument("--policy_calibration", type=str, default="percentile", 
                     choices=["raw", "percentile", "selective"],
                     help="Calibration method for policy scores: raw (simple), percentile (normalized), selective (conservative)")
    ap.add_argument("--threshold", type=float, default=None, help="Custom threshold (overrides checkpoint threshold)")
    ap.add_argument("--olid_csv", type=str, default=None, help="Optional CSV with OLID model predictions")
    ap.add_argument("--max_voting", action="store_true", help="Use Max-Voting (Logical OR) strategy")
    ap.add_argument("--out", type=str, required=True, help="Output CSV path for ensemble predictions")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
	main()


