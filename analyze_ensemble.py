"""
Analyze ensemble predictions: MMBT vs Policy RAG agreement/disagreement
and suggest optimal threshold.
"""
import pandas as pd
import numpy as np
import argparse
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def analyze_ensemble(preds_csv, out_dir="runs/mmbt_clip_b32"):
    """Analyze ensemble predictions and suggest improvements."""
    df = pd.read_csv(preds_csv)
    
    print("=" * 80)
    print("ENSEMBLE PREDICTION ANALYSIS")
    print("=" * 80)
    
    # Basic stats
    print(f"\n📊 Basic Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  MMBT score range: [{df['prob_model'].min():.4f}, {df['prob_model'].max():.4f}]")
    print(f"  Policy score range: [{df['policy_score'].min():.4f}, {df['policy_score'].max():.4f}]")
    print(f"  Ensemble score range: [{df['prob_final'].min():.4f}, {df['prob_final'].max():.4f}]")
    print(f"  Current threshold (from checkpoint): ~0.10-0.27 (inferred from labels)")
    
    # Correlation
    corr, p_val = pearsonr(df['prob_model'], df['policy_score'])
    print(f"\n🔗 Correlation Analysis:")
    print(f"  Pearson correlation (MMBT vs Policy): {corr:.4f} (p={p_val:.2e})")
    if corr < 0.3:
        print("  ⚠️  Low correlation: Models are learning different signals (good for diversity)")
    elif corr < 0.7:
        print("  ✓ Moderate correlation: Some agreement, some complementarity")
    else:
        print("  ⚠️  High correlation: Models may be redundant")
    
    # Agreement/disagreement
    # Define "high" as >0.5, "low" as <0.3
    df['mmbt_high'] = df['prob_model'] > 0.5
    df['mmbt_low'] = df['prob_model'] < 0.3
    df['policy_high'] = df['policy_score'] > 0.5
    df['policy_low'] = df['policy_score'] < 0.3
    
    both_high = ((df['mmbt_high']) & (df['policy_high'])).sum()
    both_low = ((df['mmbt_low']) & (df['policy_low'])).sum()
    mmbt_high_policy_low = ((df['mmbt_high']) & (df['policy_low'])).sum()
    mmbt_low_policy_high = ((df['mmbt_low']) & (df['policy_high'])).sum()
    
    print(f"\n🤝 Agreement/Disagreement Breakdown:")
    print(f"  Both HIGH (MMBT>0.5, Policy>0.5): {both_high} ({100*both_high/len(df):.1f}%)")
    print(f"  Both LOW (MMBT<0.3, Policy<0.3): {both_low} ({100*both_low/len(df):.1f}%)")
    print(f"  MMBT HIGH, Policy LOW: {mmbt_high_policy_low} ({100*mmbt_high_policy_low/len(df):.1f}%)")
    print(f"  MMBT LOW, Policy HIGH: {mmbt_low_policy_high} ({100*mmbt_low_policy_high/len(df):.1f}%)")
    
    # Disagreement cases
    disagreement = df[((df['mmbt_low']) & (df['policy_high'])) | 
                       ((df['mmbt_high']) & (df['policy_low']))].copy()
    disagreement['disagreement_magnitude'] = np.abs(disagreement['prob_model'] - disagreement['policy_score'])
    disagreement = disagreement.sort_values('disagreement_magnitude', ascending=False)
    
    print(f"\n⚠️  Top 10 Largest Disagreements (MMBT vs Policy):")
    for idx, row in disagreement.head(10).iterrows():
        print(f"  {row['img']}: MMBT={row['prob_model']:.4f}, Policy={row['policy_score']:.4f}, "
              f"Diff={row['disagreement_magnitude']:.4f}")
        print(f"    Text: {row['text'][:70]}...")
    
    # Threshold analysis
    print(f"\n📈 Threshold Analysis (on Ensemble Scores):")
    current_hate_count = (df['label_final'] == 1).sum()
    print(f"  Current predictions: {current_hate_count} hate ({100*current_hate_count/len(df):.1f}%)")
    
    # Suggest thresholds
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"\n  Suggested thresholds (if you had labels, compute F1/Precision/Recall):")
    for thr in thresholds:
        count = (df['prob_final'] >= thr).sum()
        pct = 100 * count / len(df)
        print(f"    thr={thr:.1f}: {count} hate predictions ({pct:.1f}%)")
    
    # Policy score distribution
    print(f"\n📊 Policy Score Distribution:")
    policy_high_count = (df['policy_score'] > 0.8).sum()
    policy_med_count = ((df['policy_score'] >= 0.5) & (df['policy_score'] <= 0.8)).sum()
    policy_low_count = (df['policy_score'] < 0.5).sum()
    print(f"  Policy > 0.8: {policy_high_count} ({100*policy_high_count/len(df):.1f}%)")
    print(f"  Policy 0.5-0.8: {policy_med_count} ({100*policy_med_count/len(df):.1f}%)")
    print(f"  Policy < 0.5: {policy_low_count} ({100*policy_low_count/len(df):.1f}%)")
    
    if policy_high_count > len(df) * 0.5:
        print("  ⚠️  WARNING: Policy RAG is flagging >50% as high-risk. May be over-sensitive.")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if mmbt_low_policy_high > len(df) * 0.2:
        print("  1. ⚠️  Policy RAG is over-flagging (many low MMBT scores get high policy scores)")
        print("     → Consider: Normalize policy scores, or increase alpha (weight MMBT more)")
    if corr < 0.2:
        print("  2. ✓ Models are complementary (low correlation = good diversity)")
    if df['prob_final'].median() < 0.3:
        print("  3. ⚠️  Ensemble scores are generally low. Current threshold may be too low.")
        print("     → Consider: Recalibrate threshold on dev set with ensemble scores")
    
    # Save detailed disagreement cases
    disagreement_out = f"{out_dir}/disagreements.csv"
    disagreement[['img', 'text', 'prob_model', 'policy_score', 'prob_final', 'label_final', 
                  'disagreement_magnitude']].to_csv(disagreement_out, index=False)
    print(f"\n💾 Saved detailed disagreements to: {disagreement_out}")
    
    # Simple scatter plot
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['prob_model'], df['policy_score'], alpha=0.5, s=20)
        plt.xlabel('MMBT Score (prob_model)')
        plt.ylabel('Policy RAG Score')
        plt.title('MMBT vs Policy RAG Scores')
        plt.grid(True, alpha=0.3)
        plt.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='Policy threshold')
        plt.axvline(0.5, color='b', linestyle='--', alpha=0.5, label='MMBT threshold')
        plt.legend()
        plot_path = f"{out_dir}/ensemble_scatter.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved scatter plot to: {plot_path}")
    except Exception as e:
        print(f"⚠️  Could not generate plot: {e}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ensemble predictions")
    parser.add_argument("--preds_csv", type=str, default="runs/mmbt_clip_b32/preds_ensemble.csv",
                        help="Path to ensemble predictions CSV")
    parser.add_argument("--out_dir", type=str, default="runs/mmbt_clip_b32",
                        help="Output directory for analysis files")
    args = parser.parse_args()
    
    analyze_ensemble(args.preds_csv, args.out_dir)

