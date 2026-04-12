import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from transformers import CLIPModel, CLIPProcessor

from data import HatefulMemes
from ensemble_infer import PolicyRAGScorer, load_ckpt_threshold
from model import MMBTCLIP
from torch.utils.data import DataLoader


def evaluate_ensemble_accuracy(
    dev_json: str,
    img_dir: str,
    ckpt_path: str,
    policy_index: str,
    alpha: float = 0.85,
    policy_calibration: str = "percentile",
    clip_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 16,
    num_workers: int = 2,
    fp16: bool = True,
    threshold: float = None,
    olid_csv: str = None,
    max_voting: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint threshold, or use provided threshold
    _, ckpt_thr = load_ckpt_threshold(ckpt_path)
    thr = threshold if threshold is not None else ckpt_thr
    print(f"[eval] Using threshold: {thr:.4f} {'(from checkpoint)' if threshold is None else '(custom)'}")
    
    # Load model
    print(f"[eval] Loading model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    dtype = torch.float16 if fp16 and torch.cuda.is_available() else torch.float32
    clip_model = CLIPModel.from_pretrained(clip_name, torch_dtype=dtype)
    # Check if LoRA was used (we'll infer from checkpoint structure)
    use_lora = any("lora" in k for k in ckpt["model"].keys())
    model = MMBTCLIP(clip_model, proj_dim=256, use_lora=use_lora).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    
    # Load dev set
    print(f"[eval] Loading dev set from {dev_json}...")
    processor = CLIPProcessor.from_pretrained(clip_name)
    dev_dataset = HatefulMemes(img_dir, dev_json, processor, split="dev", aug=False)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Get MMBT predictions
    print("[eval] Getting MMBT predictions...")
    texts_list = []
    labels_list = []
    mmbt_probs = []
    
    with torch.no_grad():
        for batch in dev_loader:
            labels = batch.pop("labels").numpy()
            meta = batch.pop("meta", None)
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            with torch.amp.autocast(device_type="cuda", enabled=(fp16 and device == "cuda")):
                logits = model(**batch)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
            
            # Extract texts from meta
            if isinstance(meta, dict):
                length = len(next(iter(meta.values())))
                for i in range(length):
                    texts_list.append(meta["text"][i])
            else:
                texts_list.extend([m["text"] for m in meta])
            
            labels_list.extend(labels.tolist())
            mmbt_probs.extend(probs.tolist())
    
    y_true = np.array(labels_list)
    p_mmbt = np.array(mmbt_probs).flatten()
    
    
    # Get policy RAG scores
    print("[eval] Getting policy RAG scores...")
    rag = PolicyRAGScorer(policy_index, device=device)
    p_policy = rag.score(texts_list, calibration=policy_calibration)

    # Get OLID scores (optional)
    p_olid = None
    if olid_csv:
        print(f"[eval] Loading OLID predictions from {olid_csv}...")
        olid_df = pd.read_csv(olid_csv)
        # Check if length matches
        if len(olid_df) != len(y_true):
             print(f"WARNING: OLID CSV length ({len(olid_df)}) != Dev Set length ({len(y_true)})")
             # Proceed assuming the first N match or crash?
             # Let's slice/pad to be safe or just use what we have if larger
             # If smaller, we have a problem.
             if len(olid_df) < len(y_true):
                 raise ValueError("OLID CSV has fewer rows than dev set!")
             olid_df = olid_df.iloc[:len(y_true)]
        
        if "olid_prob" in olid_df.columns:
            p_olid = olid_df["olid_prob"].astype(float).values
        elif "prob" in olid_df.columns:
             p_olid = olid_df["prob"].astype(float).values
        else:
             raise ValueError("OLID CSV missing 'olid_prob' or 'prob' column")
    
    # Ensemble
    if max_voting:
        # Max-Voting Logic
        print(f"[eval] Ensemble Strategy: Max-Voting (Logical OR)")
        l_mmbt = (p_mmbt >= thr).astype(int)
        
        policy_thr = thr if 0.3 <= thr <= 0.7 else 0.5
        l_policy = (p_policy >= policy_thr).astype(int)
        
        if p_olid is not None:
             l_olid = (p_olid >= 0.5).astype(int)
             # Logical OR
             y_tmp = np.maximum(l_mmbt, l_policy)
             y_pred = np.maximum(y_tmp, l_olid)
             
             # Composite score for AUC (Max Probability)
             p_final = np.maximum(p_mmbt, p_policy)
             p_final = np.maximum(p_final, p_olid)
        else:
             y_pred = np.maximum(l_mmbt, l_policy)
             p_final = np.maximum(p_mmbt, p_policy)
             
    else:
        # Weighted Average
        if p_olid is not None:
            remain = 1.0 - alpha
            p_final = alpha * p_mmbt + (remain / 2) * p_policy + (remain / 2) * p_olid
            print(f"[eval] Ensemble Components: MMBT ({alpha}), Policy ({remain/2:.2f}), OLID ({remain/2:.2f})")
        else:
            p_final = alpha * p_mmbt + (1.0 - alpha) * p_policy
            print(f"[eval] Ensemble Components: MMBT ({alpha}), Policy ({1.0-alpha:.2f})")
            
        y_pred = (p_final >= thr).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, p_final)
    ap = average_precision_score(y_true, p_final)
    cm = confusion_matrix(y_true, y_pred)
    
    # Print results
    print("\n" + "="*70)
    print("ENSEMBLE ACCURACY EVALUATION (Dev Set)")
    print("="*70)
    print(f"Configuration:")
    print(f"  Alpha (MMBT weight): {alpha:.2f}")
    print(f"  Policy calibration: {policy_calibration}")
    print(f"  Threshold: {thr:.4f}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  PR-AUC:    {ap:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Not Hate  Hate")
    print(f"  Actual Not Hate  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"         Hate      {cm[1,0]:4d}  {cm[1,1]:4d}")
    print(f"\nBreakdown:")
    print(f"  True Negatives:  {cm[0,0]}")
    print(f"  False Positives: {cm[0,1]}")
    print(f"  False Negatives: {cm[1,0]}")
    print(f"  True Positives:  {cm[1,1]}")
    print(f"\nScore Statistics:")
    print(f"  MMBT score range: [{p_mmbt.min():.4f}, {p_mmbt.max():.4f}]")
    print(f"  Policy score range: [{p_policy.min():.4f}, {p_policy.max():.4f}]")
    print(f"  Ensemble score range: [{p_final.min():.4f}, {p_final.max():.4f}]")
    print("="*70)
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "ap": ap,
        "confusion_matrix": cm,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate ensemble accuracy on dev set")
    ap.add_argument("--dev_json", type=str, required=True, help="Dev set JSONL path")
    ap.add_argument("--img_dir", type=str, required=True, help="Image directory")
    ap.add_argument("--ckpt", type=str, required=True, help="Model checkpoint path (best.pt)")
    ap.add_argument("--policy_index", type=str, required=True, help="Policy index path")
    ap.add_argument("--alpha", type=float, default=0.85, help="MMBT weight in ensemble")
    ap.add_argument("--policy_calibration", type=str, default="percentile", 
                    choices=["raw", "percentile", "selective"],
                    help="Policy score calibration method")
    ap.add_argument("--clip", type=str, default="openai/clip-vit-base-patch32", help="CLIP model name")
    ap.add_argument("--batch", type=int, default=16, help="Batch size")
    ap.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    ap.add_argument("--fp16", action="store_true", help="Use fp16")
    ap.add_argument("--threshold", type=float, default=None, help="Custom threshold (overrides checkpoint threshold)")
    ap.add_argument("--olid_csv", type=str, default=None, help="CSV with OLID predictions for dev set")
    ap.add_argument("--max_voting", action="store_true", help="Use Max-Voting strategy")
    args = ap.parse_args()
    
    evaluate_ensemble_accuracy(
        dev_json=args.dev_json,
        img_dir=args.img_dir,
        ckpt_path=args.ckpt,
        policy_index=args.policy_index,
        alpha=args.alpha,
        policy_calibration=args.policy_calibration,
        clip_name=args.clip,
        batch_size=args.batch,
        num_workers=args.num_workers,
        fp16=args.fp16,
        threshold=args.threshold,
        olid_csv=args.olid_csv,
        max_voting=args.max_voting,
    )


if __name__ == "__main__":
    main()

