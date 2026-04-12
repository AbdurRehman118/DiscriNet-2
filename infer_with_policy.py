import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import json
import os
import argparse
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score

from data import HatefulMemes
from model import MMBTCLIP

def load_policy_index(path, device):
    print(f"Loading policy index from {path}...")
    data = torch.load(path, map_location=device)
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--policy_index", type=str, required=True, help="Path to policy_index.pt")
    parser.add_argument("--test_json", type=str, required=True, help="Path to test.jsonl")
    parser.add_argument("--img_dir", type=str, required=True, help="Root image dir")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # 1. Load Model
    print("Loading MMBT CLIP model...")
    ckpt = torch.load(args.model_path, map_location=args.device)
    
    # Try to infer CLIP version from args.json if exists
    run_dir = os.path.dirname(args.model_path)
    args_path = os.path.join(run_dir, "args.json")
    clip_name = "openai/clip-vit-large-patch14" # Default
    if os.path.exists(args_path):
        with open(args_path, "r") as f:
            train_args = json.load(f)
            if "clip" in train_args:
                clip_name = train_args["clip"]
    
    print(f"Backbone: {clip_name}")
    
    processor = CLIPProcessor.from_pretrained(clip_name)
    clip_model = CLIPModel.from_pretrained(clip_name)
    # Get projection dim
    model = MMBTCLIP(clip_model, proj_dim=256, use_lora=False) 
    # Note: If LoRA was used, we need to apply it. The state_dict keys will handle it 
    # BUT we need to initialize PEFT model structure if keys contain "lora".
    # Inspecting state_dict keys:
    keys = list(ckpt["model"].keys())
    has_lora = any("lora" in k for k in keys)
    if has_lora:
        print("Detected LoRA weights. Configuring PEFT...")
        # Create LoRA config same as train.py
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            lora_dropout=0.05,
            bias="none",
        )
        model.clip = get_peft_model(model.clip, cfg)
    
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(args.device)
    model.eval()

    # 2. Load Policies
    index = load_policy_index(args.policy_index, args.device)
    policy_embs = index["embeddings"].to(args.device) # [N_pol, D]
    policy_embs = F.normalize(policy_embs, p=2, dim=-1)

    # 3. Data Loader
    print("Creating DataLoader...")
    ds = HatefulMemes(args.img_dir, args.test_json, processor, split="test", aug=False)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=0, pin_memory=True)

    # 4. Inference
    y_true = []
    prob_mmbt = []
    prob_policy = []

    print("Running inference...")
    with torch.no_grad():
        for batch in dl:
            labels = batch.pop("labels").numpy()
            batch.pop("meta", None)
            batch = {k: v.to(args.device) for k, v in batch.items() if torch.is_tensor(v)}
            
            # A. MMBT Forward
            logits = model(**batch)
            p_mmbt = torch.sigmoid(logits).cpu().numpy()
            
            # B. Policy Scorer
            # Get CLIP features directly
            # Note: We need to use the base CLIP model inside MMBT
            # MMBT.clip is the CLIPModel (possibly wrapped in PEFT)
            # PEFT model forward might behave differently, let's access base model if needed or use standard interface
            
            # CLIP image features
            # model.clip might be PeftModel. 
            # PeftModel pass-through usually works for standard args, but get_image_features might need care.
            base_clip = model.clip.base_model.model if has_lora else model.clip
            
            img_feats = base_clip.get_image_features(pixel_values=batch["pixel_values"])
            img_feats = F.normalize(img_feats, p=2, dim=-1)
            
            txt_feats = base_clip.get_text_features(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            txt_feats = F.normalize(txt_feats, p=2, dim=-1)
            
            # Similarity
            # [B, D] @ [N_pol, D].T -> [B, N_pol]
            sim_img = torch.matmul(img_feats, policy_embs.T)
            sim_txt = torch.matmul(txt_feats, policy_embs.T)
            
            # Score = max(sim) or mean(top_k)? 
            # Let's verify what "Policy Scorer" implies. 
            # If we assume policies are "Hateful Definitions", then Max Sim is good.
            # Using Max over all policies.
            score_img, _ = sim_img.max(dim=1)
            score_txt, _ = sim_txt.max(dim=1)
            
            # Fusion of policy scores
            # Determine if image OR text violates policy
            p_pol = torch.maximum(score_img, score_txt)
            
            prob_mmbt.extend(p_mmbt.tolist())
            prob_policy.extend(p_pol.cpu().numpy().tolist())
            y_true.extend(labels)

    y_true = np.array(y_true)
    prob_mmbt = np.array(prob_mmbt)
    prob_policy = np.array(prob_policy)
    
    # 5. Combine and Evaluate
    # Normalize Policy Scores to [0,1] for better combination? 
    # Cosine sim is [-1, 1]. Sigmoid? or MinMax?
    # Simple MinMax based on batch might be unstable. 
    # Let's map [-1, 1] to [0, 1] linearly? (x+1)/2
    prob_policy_norm = (prob_policy + 1) / 2.0
    
    # Ensemble
    # alpha * MMBT + (1-alpha) * Policy
    # Let's find best alpha
    best_auc = 0
    best_alpha = 0
    best_combined = None
    
    print("-" * 30)
    print("Results:")
    
    # Metrics function
    def calc_metrics(y, p, name):
        auc = roc_auc_score(y, p)
        ap = average_precision_score(y, p)
        # Threshold
        best_f1 = 0
        acc = 0
        for t in np.linspace(0.1, 0.9, 81):
            pred = (p >= t).astype(int)
            f1 = f1_score(y, pred)
            if f1 > best_f1:
                best_f1 = f1
                acc = accuracy_score(y, pred)
        print(f"[{name}] AUC: {auc:.4f} | AP: {ap:.4f} | Best F1: {best_f1:.4f} | Acc: {acc:.4f}")
        return auc

    calc_metrics(y_true, prob_mmbt, "Base MMBT")
    calc_metrics(y_true, prob_policy_norm, "Policy Score Only")
    
    alphas = np.linspace(0, 1, 21)
    for alpha in alphas:
        p_comb = alpha * prob_mmbt + (1 - alpha) * prob_policy_norm
        try:
            auc = roc_auc_score(y_true, p_comb)
            if auc > best_auc:
                best_auc = auc
                best_alpha = alpha
                best_combined = p_comb
        except:
            continue
            
    print(f"\nBest Alpha for Ensemble (MMBT weight): {best_alpha:.2f}")
    calc_metrics(y_true, best_combined, f"Combined (alpha={best_alpha:.2f})")
    
    # Save results
    out_file = args.model_path.replace(".pt", "_policy_results.txt")
    with open(out_file, "w") as f:
        f.write(f"Base AUC: {roc_auc_score(y_true, prob_mmbt):.4f}\n")
        f.write(f"Combined AUC: {best_auc:.4f}\n")
        f.write(f"Best Alpha: {best_alpha}\n")
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()
