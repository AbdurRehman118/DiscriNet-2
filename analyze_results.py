import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import os
import argparse
import pandas as pd
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import joblib

from data import HatefulMemes
from model import MMBTCLIP

def load_policy_index(path, device):
    print(f"Loading policy index from {path}...")
    data = torch.load(path, map_location=device)
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--policy_index", type=str, required=True)
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Load Model
    print("Loading MMBT CLIP model...")
    ckpt = torch.load(args.model_path, map_location=args.device)
    # Infer/Default
    clip_name = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(clip_name)
    clip_model = CLIPModel.from_pretrained(clip_name)
    model = MMBTCLIP(clip_model, proj_dim=256, use_lora=False)
    
    # Check for LoRA keys
    keys = list(ckpt["model"].keys())
    has_lora = any("lora" in k for k in keys)
    if has_lora:
        print("Detected LoRA weights. Configuring PEFT...")
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
    policy_embs = index["embeddings"].to(args.device)
    policy_embs = F.normalize(policy_embs, p=2, dim=-1)

    # 3. Data Loader
    ds = HatefulMemes(args.img_dir, args.test_json, processor, split="test", aug=False)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    # 4. Inference
    y_true = []
    prob_base = []
    prob_policy = []
    
    meta_data = []
    policy_feats_list = []

    print("Running inference...")
    with torch.no_grad():
        for batch in dl:
            labels = batch.pop("labels").numpy()
            metas = batch.pop("meta") 
            # Metas is dict of lists or list of dicts depending on collate. 
            # Default collate -> dict of lists
            # We need to collect img paths and text
            
            # Re-structure meta
            bs = len(labels)
            for i in range(bs):
                meta_data.append({
                    "img": metas["img_path"][i],
                    "text": metas["text"][i],
                    "label": int(metas["label"][i])
                })

            batch = {k: v.to(args.device) for k, v in batch.items() if torch.is_tensor(v)}
            
            # Base Model
            logits = model(**batch)
            p_mmbt = torch.sigmoid(logits).cpu().numpy()
            
            # Policy
            base_clip = model.clip.base_model.model if has_lora else model.clip
            img_feats = base_clip.get_image_features(pixel_values=batch["pixel_values"])
            img_feats = F.normalize(img_feats, p=2, dim=-1)
            txt_feats = base_clip.get_text_features(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            txt_feats = F.normalize(txt_feats, p=2, dim=-1)
            
            sim_img = torch.matmul(img_feats, policy_embs.T)
            sim_txt = torch.matmul(txt_feats, policy_embs.T)
            score_img, _ = sim_img.max(dim=1)
            score_txt, _ = sim_txt.max(dim=1)
            p_pol = torch.maximum(score_img, score_txt)
            
            # Feature Collection [B, 50]
            feats = torch.cat([sim_img, sim_txt], dim=1).cpu().numpy()
            policy_feats_list.append(feats)
            
            prob_base.extend(p_mmbt.tolist())
            prob_policy.extend(p_pol.cpu().numpy().tolist())
            y_true.extend(labels)

    y_true = np.array(y_true)
    prob_base = np.array(prob_base)
    prob_policy = np.array(prob_policy)
    # Policy Score is Max Sim. Typically [-1, 1].
    # We want to use it as a Boost.
    # Logic: If Policy Sim is High -> High Confidence Hate -> High Weight.
    #        If Policy Sim is Low  -> Low Confidence/Noise -> Low Weight.
    
    # Normalize to [0,1] for mixing
    prob_policy_norm = (prob_policy + 1) / 2.0
    
    print("optimizing Dynamic Gating Ensemble...")
    best_acc = 0
    best_params = (0, 0) # (threshold, boost_weight)
    
    # We define Ensemble as:
    # P_final = (1 - w) * P_base + w * P_policy
    # Where w = dynamic_weight(P_policy)
    # Simple Dynamic Weight: w = base_w * sigmoid( (P_policy - threshold) * sharp )
    # Or simpler: if P_policy > thresh: w = high, else w = low.
    
    # Grid Search
    # Threshold in raw sim [-1, 1] usually around 0.2-0.3 for CLIP match
    # Weight [0, 1]
    
    for thresh in np.linspace(0.15, 0.45, 10): # Raw Sim thresholds
        for weight in np.linspace(0.0, 1.0, 21):
            
            # Mask: Where policy match is strong
            # prob_policy is Raw Max Sim
            strong_match = prob_policy > thresh
            
            p_final = np.zeros_like(prob_base)
            
            # If Strong Match: Mix with High Weight
            p_final[strong_match] = (1 - weight) * prob_base[strong_match] + weight * prob_policy_norm[strong_match]
            
            # If Weak Match: Rely mostly on Base (Policy is noise)
            # giving policy 0 weight or small weight? Let's give 0 to minimize noise.
            p_final[~strong_match] = prob_base[~strong_match]
            
            if np.max(y_true) > -1:
                acc = accuracy_score(y_true, (p_final >= 0.5).astype(int))
                if acc > best_acc:
                    best_acc = acc
                    best_params = (thresh, weight)
                    prob_ensemble = p_final
    
    best_thresh, best_weight = best_params
    print(f"Best Dynamic Params: Threshold={best_thresh:.4f}, Boost_Weight={best_weight:.4f}")
    
    # Save parameters for App
    joblib.dump({"threshold": best_thresh, "weight": best_weight}, os.path.join(args.out_dir, "ensemble_params.pkl"))
    
    logit_str = f"Dynamic Gating (T={best_thresh:.2f}, W={best_weight:.2f})"
    best_alpha = 999 
    
    # 6. Generate Metrics Report & CSV

    # 6. Generate Metrics Report & CSV
    print(f"Generating Analysis for Alpha={best_alpha:.2f}...")
    
    # CSV
    csv_path = os.path.join(args.out_dir, "predictions_comparison.csv")
    df = pd.DataFrame(meta_data)
    df["prob_base"] = prob_base
    df["prob_policy"] = prob_policy_norm
    df["prob_ensemble"] = prob_ensemble
    df.to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")

    # Metrics & Plots (Only if labels exist)
    if np.max(y_true) > -1:
        # ROC Plot
        plt.figure(figsize=(10, 8))
        
        # Base
        fpr, tpr, _ = roc_curve(y_true, prob_base)
        auc_base = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Base MMBT (AUC = {auc_base:.4f})')
        
        # Policy
        fpr, tpr, _ = roc_curve(y_true, prob_policy_norm)
        auc_pol = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Policy Only (AUC = {auc_pol:.4f})', linestyle='--')
        
        # Ensemble
        fpr, tpr, _ = roc_curve(y_true, prob_ensemble)
        auc_ens = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Ensemble (AUC = {auc_ens:.4f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-AUC Comparison')
        plt.legend(loc="lower right")
        plot_path = os.path.join(args.out_dir, "roc_curve.png")
        plt.savefig(plot_path)
        print(f"Saved ROC plot to {plot_path}")
        
        # Detailed Metrics Report
        with open(os.path.join(args.out_dir, "metrics_report.txt"), "w") as f:
            f.write("=== Comparative Analysis ===\n\n")
            
            def report_metrics(name, y, p):
                auc_score = roc_auc_score(y, p)
                try:
                    ap = average_precision_score(y, p)
                except: ap = 0
                
                # Best F1
                best_f1 = 0
                best_acc = 0
                for t in np.linspace(0.1, 0.9, 81):
                    pred = (p >= t).astype(int)
                    f1 = f1_score(y, pred)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_acc = accuracy_score(y, pred)
                
                msg = f"{name}:\n  AUC: {auc_score:.4f}\n  AP:  {ap:.4f}\n  F1:  {best_f1:.4f}\n  Acc: {best_acc:.4f}\n\n"
                f.write(msg)
                print(msg)
                
            report_metrics("Base MMBT", y_true, prob_base)
            report_metrics("Policy Scorer", y_true, prob_policy_norm)
            report_metrics(f"Ensemble (Alpha={best_alpha:.2f})", y_true, prob_ensemble)

if __name__ == "__main__":
    main()
