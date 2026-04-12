import pandas as pd
import numpy as np

df = pd.read_csv('runs/mmbt_clip_b32/preds_ensemble_final.csv')

print("="*70)
print("FINAL ENSEMBLE PREDICTIONS SUMMARY (Test Set)")
print("="*70)
print(f"\nTotal samples: {len(df)}")

# Prediction breakdown
num_hate = (df["label_final"] == 1).sum()
num_not_hate = (df["label_final"] == 0).sum()
print(f"\nPredictions:")
print(f"  Hate (label_final=1):     {num_hate:4d} ({num_hate/len(df)*100:.1f}%)")
print(f"  Not Hate (label_final=0):  {num_not_hate:4d} ({num_not_hate/len(df)*100:.1f}%)")

# Score statistics
print(f"\nScore Statistics:")
print(f"  MMBT score range:     [{df['prob_model'].min():.4f}, {df['prob_model'].max():.4f}]")
print(f"  Policy score range:   [{df['policy_score'].min():.4f}, {df['policy_score'].max():.4f}]")
print(f"  Ensemble score range: [{df['prob_final'].min():.4f}, {df['prob_final'].max():.4f}]")

# Confidence levels
high_conf = (df["prob_final"] > 0.5).sum()
med_conf = ((df["prob_final"] >= 0.15) & (df["prob_final"] <= 0.5)).sum()
low_conf = (df["prob_final"] < 0.15).sum()

print(f"\nConfidence Distribution (Ensemble Scores):")
print(f"  High confidence (>0.5):     {high_conf:4d} ({high_conf/len(df)*100:.1f}%)")
print(f"  Medium confidence (0.15-0.5): {med_conf:4d} ({med_conf/len(df)*100:.1f}%)")
print(f"  Low confidence (<0.15):     {low_conf:4d} ({low_conf/len(df)*100:.1f}%)")

# Agreement between MMBT and Policy
mmbt_high = df["prob_model"] > 0.5
policy_high = df["policy_score"] > 0.5
both_high = (mmbt_high & policy_high).sum()
both_low = ((~mmbt_high) & (~policy_high)).sum()
mmbt_high_policy_low = (mmbt_high & ~policy_high).sum()
mmbt_low_policy_high = (~mmbt_high & policy_high).sum()

print(f"\nMMBT vs Policy Agreement:")
print(f"  Both HIGH (MMBT>0.5, Policy>0.5):  {both_high:4d} ({both_high/len(df)*100:.1f}%)")
print(f"  Both LOW (MMBT<0.5, Policy<0.5):   {both_low:4d} ({both_low/len(df)*100:.1f}%)")
print(f"  MMBT HIGH, Policy LOW:              {mmbt_high_policy_low:4d} ({mmbt_high_policy_low/len(df)*100:.1f}%)")
print(f"  MMBT LOW, Policy HIGH:              {mmbt_low_policy_high:4d} ({mmbt_low_policy_high/len(df)*100:.1f}%)")

# Top predictions
print(f"\nTop 10 Highest Ensemble Scores (Most Likely Hate):")
top_hate = df.nlargest(10, "prob_final")[["img", "text", "prob_model", "policy_score", "prob_final"]]
for idx, row in top_hate.iterrows():
    text_short = row["text"][:60] + "..." if len(row["text"]) > 60 else row["text"]
    print(f"  {row['img']:12s} | {text_short:60s} | Final={row['prob_final']:.3f}")

print(f"\nTop 10 Lowest Ensemble Scores (Most Likely Not Hate):")
top_not_hate = df.nsmallest(10, "prob_final")[["img", "text", "prob_model", "policy_score", "prob_final"]]
for idx, row in top_not_hate.iterrows():
    text_short = row["text"][:60] + "..." if len(row["text"]) > 60 else row["text"]
    print(f"  {row['img']:12s} | {text_short:60s} | Final={row['prob_final']:.3f}")

print("="*70)

