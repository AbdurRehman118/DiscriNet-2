import argparse
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

from eval_ensemble_accuracy import evaluate_ensemble_accuracy


def find_optimal_threshold(
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
):
    """Find optimal threshold by evaluating multiple thresholds on dev set."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from data import HatefulMemes
    from ensemble_infer import PolicyRAGScorer, load_ckpt_threshold
    from model import MMBTCLIP
    from torch.utils.data import DataLoader
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and get predictions (same as eval script)
    _, _ = load_ckpt_threshold(ckpt_path)
    
    print(f"[threshold] Loading model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    dtype = torch.float16 if fp16 and torch.cuda.is_available() else torch.float32
    clip_model = CLIPModel.from_pretrained(clip_name, torch_dtype=dtype)
    use_lora = any("lora" in k for k in ckpt["model"].keys())
    model = MMBTCLIP(clip_model, proj_dim=256, use_lora=use_lora).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    
    processor = CLIPProcessor.from_pretrained(clip_name)
    dev_dataset = HatefulMemes(img_dir, dev_json, processor, split="dev", aug=False)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Get predictions
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
    
    rag = PolicyRAGScorer(policy_index, device=device)
    p_policy = rag.score(texts_list, calibration=policy_calibration)
    
    p_final = alpha * p_mmbt + (1.0 - alpha) * p_policy
    
    # Try different thresholds
    thresholds = np.arange(0.1, 0.95, 0.05)
    results = []
    
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION (Dev Set)")
    print("="*80)
    print(f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-"*80)
    
    best_f1 = -1
    best_thr = 0.5
    best_metrics = {}
    
    for thr in thresholds:
        y_pred = (p_final >= thr).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results.append({
            "threshold": thr,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })
        
        print(f"{thr:<10.2f} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    
    print("="*80)
    print(f"\n🎯 BEST THRESHOLD: {best_thr:.3f} (F1={best_f1:.4f})")
    print(f"   Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"   Precision: {best_metrics['precision']:.4f}")
    print(f"   Recall:    {best_metrics['recall']:.4f}")
    print(f"   F1-Score:  {best_metrics['f1']:.4f}")
    
    # Plot
    results = np.array([(r["threshold"], r["accuracy"], r["precision"], r["recall"], r["f1"]) 
                        for r in results])
    plt.figure(figsize=(10, 6))
    plt.plot(results[:, 0], results[:, 1], label="Accuracy", marker="o")
    plt.plot(results[:, 0], results[:, 2], label="Precision", marker="s")
    plt.plot(results[:, 0], results[:, 3], label="Recall", marker="^")
    plt.plot(results[:, 0], results[:, 4], label="F1-Score", marker="d")
    plt.axvline(best_thr, color="r", linestyle="--", label=f"Best (F1) @ {best_thr:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold Optimization (Alpha={alpha}, Calibration={policy_calibration})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = "runs/mmbt_clip_b32/threshold_optimization.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n💾 Saved plot → {out_path}")
    
    return best_thr, best_metrics


def main():
    ap = argparse.ArgumentParser(description="Find optimal threshold for ensemble")
    ap.add_argument("--dev_json", type=str, required=True)
    ap.add_argument("--img_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--policy_index", type=str, required=True)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--policy_calibration", type=str, default="percentile", 
                    choices=["raw", "percentile", "selective"])
    ap.add_argument("--clip", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()
    
    find_optimal_threshold(
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
    )


if __name__ == "__main__":
    main()

