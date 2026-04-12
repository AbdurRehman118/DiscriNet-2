import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import os
import argparse
import joblib
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from data import HatefulMemes

def load_policy_index(path, device):
    print(f"Loading policy index from {path}...")
    data = torch.load(path, map_location=device)
    return data

def extract_features(model, dl, policy_embs, device):
    all_feats = []
    all_labels = []
    
    # Access CLIP Model
    # input 'model' is Namespace(clip=clip_model)
    clip_model = model.clip
    
    # If using HF CLIPModel (standard)
    # No .base_model indirection needed unless it's PEFT
    # Just check if it has visual/text_model or use it directly
    if hasattr(clip_model, "get_image_features"):
        base_clip = clip_model
    else:
         # Fallback/PEFT
        base_clip = clip_model.base_model.model if hasattr(clip_model, "base_model") else clip_model

    with torch.no_grad():
        for batch in tqdm(dl, desc="Extracting Features"):
            labels = batch.pop("labels").numpy()
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            
            # Extract Embeddings
            img_feats = F.normalize(base_clip.get_image_features(pixel_values=batch["pixel_values"]), p=2, dim=-1)
            txt_feats = F.normalize(base_clip.get_text_features(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]), p=2, dim=-1)
            
            # Similarity to Policies [B, N_Policies]
            sim_img = torch.matmul(img_feats, policy_embs.T)
            sim_txt = torch.matmul(txt_feats, policy_embs.T)
            
            # Concatenate [B, 2*N_Policies]
            feats = torch.cat([sim_img, sim_txt], dim=1).cpu().numpy()
            
            all_feats.append(feats)
            all_labels.extend(labels)
            
    return np.vstack(all_feats), np.array(all_labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--dev_json", type=str, required=True)
    parser.add_argument("--policy_index", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/supervised_policy")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Load CLIP (Vanilla is fine for feature extraction, or use trained backbone?)
    # Using Vanilla CLIP-Large is standardized. Using Trained Backbone is better (aligned).
    # Let's use Vanilla for "Policy Alignment" as policies were encoded with Vanilla.
    # Actually, policies were encoded with Vanilla CLIP. So we should use Vanilla CLIP for queries too.
    
    print("Loading CLIP...")
    clip_name = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(clip_name)
    clip_model = CLIPModel.from_pretrained(clip_name).to(args.device)
    clip_model.eval()
    
    # Mock Wrapper for data loader compatibility (if needed) or just access clip directly
    # Dataset needs processor
    
    # 2. Load Policies
    policy_data = torch.load(args.policy_index, map_location=args.device)
    policy_embs = F.normalize(policy_data["embeddings"].to(args.device), p=2, dim=-1)
    
    # 3. Extract Train Features
    print("Processing Train Set...")
    ds_train = HatefulMemes(args.img_dir, args.train_json, processor, split="train", aug=False)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, num_workers=0) # Windows
    X_train, y_train = extract_features(argparse.Namespace(clip=clip_model), dl_train, policy_embs, args.device)
    
    # 4. Extract Dev Features
    print("Processing Dev Set...")
    ds_dev = HatefulMemes(args.img_dir, args.dev_json, processor, split="dev", aug=False)
    dl_dev = DataLoader(ds_dev, batch_size=args.batch_size, num_workers=0)
    X_dev, y_dev = extract_features(argparse.Namespace(clip=clip_model), dl_dev, policy_embs, args.device)
    
    # 5. Train Supervised Classifier
    print(f"Training Classifier on {X_train.shape[1]} features...")
    # RandomForest is robust
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # 6. Evaluate
    y_prob_dev = clf.predict_proba(X_dev)[:, 1]
    auc = roc_auc_score(y_dev, y_prob_dev)
    acc = accuracy_score(y_dev, (y_prob_dev >= 0.5).astype(int))
    
    print(f"\nSupervised Policy Scorer Results (Dev):")
    print(f"AUC: {auc:.4f}")
    print(f"Acc: {acc:.4f}")
    
    # Save Model
    joblib.dump(clf, os.path.join(args.out_dir, "policy_classifier.pkl"))
    
    # Save Feature Data for Ensemble Integration
    # We need to save X_dev predictions for the Stacking step?
    # Actually, we can just save the model and let the stacking script run it.
    
    # 7. Analyze Feature Importance
    imports = clf.feature_importances_
    # First 25 are Image-Policy, Next 25 are Text-Policy
    # Load Policy Texts
    pol_texts = policy_data["texts"]
    n_p = len(pol_texts)
    
    print("\nTop 5 Important Policies (Image Trigger):")
    img_imps = imports[:n_p]
    top_img = np.argsort(img_imps)[::-1][:5]
    for i in top_img:
        print(f"  {img_imps[i]:.4f}: {pol_texts[i][:60]}...")
        
    print("\nTop 5 Important Policies (Text Trigger):")
    txt_imps = imports[n_p:]
    top_txt = np.argsort(txt_imps)[::-1][:5]
    for i in top_txt:
        print(f"  {txt_imps[i]:.4f}: {pol_texts[i][:60]}...")

if __name__ == "__main__":
    main()
