
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score

def analyze_mmbt(path):
    print(f"\n=== MMBT-CLIP Results ({path}) ===")
    if not os.path.exists(path):
        print("File not found.")
        return
    df = pd.read_csv(path)
    if 'label_final' in df.columns and 'prob_final' in df.columns:
        y_true = df['label_final']
        y_score = df['prob_final']
        y_pred = (y_score >= 0.5).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_score)
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC:  {auc:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
    else:
        print("Labels not found, showing distribution:")
        print(df['prob_final'].describe())

def analyze_fb_memes(path):
    print(f"\n=== FB Memes ViT Large Results ({path}) ===")
    if not os.path.exists(path):
        print("File not found.")
        return
    df = pd.read_csv(path)
    print("Prediction Distribution (Probability of Hate):")
    print(df['prob'].describe())
    print("\nPredicted Classes (Threshold 0.5):")
    print((df['prob'] >= 0.5).value_counts())

def analyze_mmhs(path):
    print(f"\n=== MMHS150K ViT Large Results ({path}) ===")
    if not os.path.exists(path):
        print("File not found.")
        return
    cm = np.load(path)
    # cm format: [[TN, FP], [FN, TP]] usually, but let's check standard sklearn output
    # sklearn confusion_matrix(y_true, y_pred) returns [[TN, FP], [FN, TP]]
    tn, fp = cm[0]
    fn, tp = cm[1]
    
    total = np.sum(cm)
    acc = (tn + tp) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

print("Calculating Metrics...")
analyze_mmbt('runs/mmbt_clip_b32/preds_ensemble_final.csv')
analyze_fb_memes('runs/fb_memes_vit_large/preds_test.csv')
analyze_mmhs('runs/mmhs150k_vit_large/confusion.npy')
