"""
H5: Sentiment as Leading Indicator
Tests whether social media sentiment predicts next-day BIST100 returns

Author: Simla Tükenmez
Student ID: 32613
Course: DSA210 - Fall 2025-2026

Hypothesis: Yesterday's sentiment (Ekşi Sözlük) predicts today's returns
(Sentiment has LEADING predictive power)

Method: Lagged regression
Y = BIST100_Return[t]
X = Sentiment_Lag1[t-1]
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load master dataset with sentiment"""
    print("Loading data...")
    df = pd.read_excel('master_data_with_sentiment.xlsx')
    df['date'] = pd.to_datetime(df['date'])
    print(f"   ✅ {len(df)} rows loaded\n")
    return df

def h5_sentiment_prediction(df):
    """Test H5: Sentiment predicts next-day returns"""
    
    print("="*80)
    print("H5: SENTIMENT AS LEADING INDICATOR")
    print("="*80)
    
    # Prepare data
    analysis_df = df[['date', 'Sentiment_Score', 'Sentiment_Lag1', 'BIST100_Return']].dropna()
    
    print(f"\nAnalysis sample: {len(analysis_df)} observations")
    print(f"   Date range: {analysis_df['date'].min().date()} to {analysis_df['date'].max().date()}")
    
    # Correlations
    print(f"\nCorrelation Analysis:")
    print("="*80)
    
    corr_same_day = analysis_df[['Sentiment_Score', 'BIST100_Return']].corr().iloc[0,1]
    corr_leading = analysis_df[['Sentiment_Lag1', 'BIST100_Return']].corr().iloc[0,1]
    
    print(f"\nSame-day (Sentiment[t] ↔ Return[t]):  r = {corr_same_day:>7.4f}")
    print(f"Leading (Sentiment[t-1] → Return[t]): r = {corr_leading:>7.4f}")
    
    if abs(corr_leading) > abs(corr_same_day):
        print(f"\n✅ Leading correlation is STRONGER")
        print(f"   → Sentiment has predictive power")
    else:
        print(f"\nSame-day correlation is stronger")
        print(f"   → Sentiment is contemporaneous, not predictive")
    
    # Regression
    print(f"\n" + "="*80)
    print("LAGGED REGRESSION ANALYSIS")
    print("="*80)
    print(f"\nModel: BIST100_Return[t] = β₀ + β₁ × Sentiment[t-1] + ε")
    
    X = analysis_df[['Sentiment_Lag1']].values
    y = analysis_df['BIST100_Return'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    beta0 = model.intercept_
    beta1 = model.coef_[0]
    r2 = model.score(X, y)
    
    # Statistical significance
    y_pred = model.predict(X)
    residuals = y - y_pred
    mse = np.sum(residuals**2) / (len(y) - 2)
    se_beta1 = np.sqrt(mse / np.sum((X - X.mean())**2))
    t_stat = beta1 / se_beta1
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - 2))
    
    print(f"\nRegression Results:")
    print(f"  β₀ (Intercept):      {beta0:>8.4f}")
    print(f"  β₁ (Sentiment[t-1]): {beta1:>8.4f} {'***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.10 else ''}")
    print(f"  SE(β₁):              {se_beta1:>8.4f}")
    print(f"  t-statistic:         {t_stat:>8.4f}")
    print(f"  P-value:             {p_value:>8.4f}")
    print(f"  R²:                  {r2:>8.4f}")
    
    # Decision
    alpha = 0.05
    print(f"\nStatistical Decision (α = {alpha}):")
    print("="*80)
    
    if p_value < alpha:
        print(f"✅ REJECT H₀")
        print(f"\nConclusion: Sentiment has SIGNIFICANT predictive power.")
        print(f"Yesterday's sentiment predicts today's returns (p = {p_value:.4f}).")
    else:
        print(f"❌ FAIL TO REJECT H₀")
        print(f"\nConclusion: Sentiment does NOT have significant predictive power.")
        print(f"Leading relationship is not statistically significant (p = {p_value:.4f}).")
    
    # Practical interpretation
    print(f"\nPractical Interpretation:")
    print("="*80)
    print(f"Current finding: r(leading) = {corr_leading:.4f}")
    
    if abs(corr_leading) < 0.05:
        print(f"→ Essentially ZERO predictive power")
        print(f"→ Ekşi Sözlük sentiment is REACTIVE, not predictive")
        print(f"→ People complain AFTER market drops, not before")
    elif abs(corr_leading) < 0.10:
        print(f"→ Very WEAK predictive power")
        print(f"→ Sentiment mostly reactive to market moves")
    else:
        print(f"→ Weak but measurable predictive power")
        print(f"→ Some forward-looking information in sentiment")
    
    # Visualization
    print(f"\nCreating visualizations...")
    visualize_h5(analysis_df, model, beta1, corr_same_day, corr_leading, p_value, r2)
    
    # Save results
    save_h5_results(analysis_df, corr_same_day, corr_leading, 
                   beta0, beta1, r2, p_value)
    
    return {
        'corr_leading': corr_leading,
        'beta1': beta1,
        'p_value': p_value,
        'r2': r2,
        'significant': p_value < alpha
    }

def visualize_h5(df, model, beta1, corr_same, corr_lead, p_value, r2):
    """Create H5 visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Scatter - Leading relationship
    ax1 = axes[0, 0]
    ax1.scatter(df['Sentiment_Lag1'], df['BIST100_Return'],
               alpha=0.4, s=30, color='#457B9D')
    
    X_range = np.linspace(df['Sentiment_Lag1'].min(), 
                         df['Sentiment_Lag1'].max(), 100).reshape(-1, 1)
    y_range = model.predict(X_range)
    ax1.plot(X_range, y_range, 'r-', linewidth=2.5, label=f'β₁={beta1:.4f}')
    
    ax1.axhline(0, color='black', linestyle=':', linewidth=1)
    ax1.axvline(0, color='black', linestyle=':', linewidth=1)
    ax1.set_xlabel('Sentiment[t-1] (Yesterday)', fontsize=11)
    ax1.set_ylabel('BIST100 Return[t] (Today, %)', fontsize=11)
    ax1.set_title('H5: Lagged Sentiment vs Returns', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.95, f'r = {corr_lead:.4f}\np = {p_value:.4f}\nR² = {r2:.4f}',
            transform=ax1.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow' if p_value < 0.05 else 'lightgray', alpha=0.8))
    
    # Plot 2: Correlation comparison
    ax2 = axes[0, 1]
    correlations = [corr_same, corr_lead]
    labels = ['Same-Day\n(Contemporaneous)', 'Leading\n(Predictive)']
    colors = ['#2E86AB', '#E63946']
    
    bars = ax2.bar(labels, correlations, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.axhline(0, color='black', linewidth=1.5)
    
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{corr:.4f}', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Correlation Coefficient', fontsize=11)
    ax2.set_title('H5: Same-Day vs Leading Correlation', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Time series - Sentiment vs Returns
    ax3 = axes[1, 0]
    
    plot_df = df.tail(200)  # Last 200 days for clarity
    ax3_twin = ax3.twinx()
    
    ax3.plot(plot_df['date'], plot_df['Sentiment_Score'],
            linewidth=1.5, color='#E63946', label='Sentiment', alpha=0.7)
    ax3_twin.plot(plot_df['date'], plot_df['BIST100_Return'],
                 linewidth=1.5, color='#2E86AB', label='BIST Return', alpha=0.7)
    
    ax3.axhline(0, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax3_twin.axhline(0, color='blue', linestyle=':', linewidth=1, alpha=0.5)
    
    ax3.set_ylabel('Sentiment Score', fontsize=11, color='#E63946')
    ax3_twin.set_ylabel('BIST100 Return (%)', fontsize=11, color='#2E86AB')
    ax3.set_title('H5: Sentiment vs Returns Over Time (Last 200 days)', 
                 fontsize=13, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor='#E63946')
    ax3_twin.tick_params(axis='y', labelcolor='#2E86AB')
    ax3.grid(True, alpha=0.3)
    
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    # Plot 4: Distribution split by sentiment direction
    ax4 = axes[1, 1]
    
    positive_sent = df[df['Sentiment_Lag1'] > 0]['BIST100_Return']
    negative_sent = df[df['Sentiment_Lag1'] <= 0]['BIST100_Return']
    
    ax4.hist(negative_sent, bins=30, alpha=0.6, color='#E63946',
            label=f'After Negative Sent\n(M={negative_sent.mean():.2f}%)', 
            edgecolor='black', linewidth=0.5)
    ax4.hist(positive_sent, bins=20, alpha=0.6, color='#2E86AB',
            label=f'After Positive Sent\n(M={positive_sent.mean():.2f}%)',
            edgecolor='black', linewidth=0.5)
    
    ax4.axvline(negative_sent.mean(), color='#E63946', linestyle='--', linewidth=2)
    ax4.axvline(positive_sent.mean(), color='#2E86AB', linestyle='--', linewidth=2)
    ax4.axvline(0, color='black', linestyle=':', linewidth=1)
    
    ax4.set_xlabel('BIST100 Return (%)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('H5: Returns Following Positive vs Negative Sentiment', 
                 fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('h5_sentiment_prediction.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: h5_sentiment_prediction.png")
    plt.close()

def save_h5_results(df, corr_same, corr_lead, beta0, beta1, r2, p_value):
    """Save H5 results"""
    
    with open('h5_results.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("H5: SENTIMENT AS LEADING INDICATOR - RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Student: Simla Tükenmez (32613)\n")
        f.write(f"Course: DSA210 - Fall 2025-2026\n\n")
        
        f.write("="*80 + "\n")
        f.write("RESEARCH QUESTION\n")
        f.write("="*80 + "\n\n")
        f.write("Does Ekşi Sözlük sentiment have LEADING predictive power for\n")
        f.write("BIST100 returns? (i.e., does yesterday's sentiment predict today's returns?)\n\n")
        
        f.write("="*80 + "\n")
        f.write("METHODOLOGY\n")
        f.write("="*80 + "\n\n")
        f.write("Model: BIST100_Return[t] = β₀ + β₁ × Sentiment[t-1] + ε\n\n")
        f.write("Sentiment Source: Ekşi Sözlük daily entries (2023-2025)\n")
        f.write("  - BERT-based Turkish sentiment model\n")
        f.write("  - Daily aggregated scores (-1 to +1)\n\n")
        f.write(f"Sample Size: {len(df)} observations\n\n")
        
        f.write("="*80 + "\n")
        f.write("CORRELATION ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Same-day (Sentiment[t] ↔ Return[t]):  r = {corr_same:>8.4f}\n")
        f.write(f"Leading (Sentiment[t-1] → Return[t]): r = {corr_lead:>8.4f}\n\n")
        
        if abs(corr_lead) > abs(corr_same):
            f.write("→ Leading correlation is STRONGER (predictive power)\n")
        else:
            f.write("→ Same-day correlation is STRONGER (reactive, not predictive)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("REGRESSION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"  β₀ (Intercept):      {beta0:>10.4f}\n")
        f.write(f"  β₁ (Sentiment[t-1]): {beta1:>10.4f} {'***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.10 else ''}\n")
        f.write(f"  P-value:             {p_value:>10.4f}\n")
        f.write(f"  R²:                  {r2:>10.4f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("CONCLUSION\n")
        f.write("="*80 + "\n\n")
        
        if p_value < 0.05:
            f.write("Decision: H5 is SUPPORTED\n\n")
            f.write(f"Sentiment has SIGNIFICANT leading predictive power (p = {p_value:.4f}).\n")
            f.write(f"Yesterday's sentiment score predicts today's returns with r = {corr_lead:.4f}.\n")
        else:
            f.write("Decision: H5 is NOT SUPPORTED\n\n")
            f.write(f"Sentiment does NOT have significant predictive power (p = {p_value:.4f}).\n")
            f.write(f"Leading correlation is essentially zero (r = {corr_lead:.4f}).\n\n")
            f.write("INTERPRETATION:\n")
            f.write("Ekşi Sözlük sentiment is REACTIVE rather than PREDICTIVE.\n")
            f.write("People express negative sentiment AFTER the market drops,\n")
            f.write("not before. Social media reflects rather than predicts market moves.\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print("   ✅ Saved: h5_results.txt")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("H5: SENTIMENT AS LEADING INDICATOR")
    print("="*80)
    print("\nDSA210 Term Project: Digital Panic and Market Dynamics")
    print("Student: Simla Tükenmez (32613)\n")
    
    df = load_data()
    results = h5_sentiment_prediction(df)
    
    print("\n" + "="*80)
    print("✅ H5 ANALYSIS COMPLETE!")
    print("="*80)
    
    print("\nOutput Files:")
    print("   1. h5_sentiment_prediction.png")
    print("   2. h5_results.txt")
    
    print("\nQuick Summary:")
    print(f"   Leading correlation: {results['corr_leading']:>8.4f}")
    print(f"   P-value:             {results['p_value']:>8.4f}")
    print(f"   R²:                  {results['r2']:>8.4f}")
    print(f"   Result:              {'✅ SIGNIFICANT' if results['significant'] else '❌ NOT SIGNIFICANT'}")
    
    if results['significant']:
        print("\n🎯 CONCLUSION: H5 SUPPORTED")
        print("   Sentiment has predictive power")
    else:
        print("\n🎯 CONCLUSION: H5 NOT SUPPORTED")
        print("   Sentiment is reactive, not predictive")
        print("   People complain AFTER market drops")
    
    print("\n")
