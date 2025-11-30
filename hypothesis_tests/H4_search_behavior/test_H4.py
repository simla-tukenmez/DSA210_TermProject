"""
H4: Google Trends - Volatility Relationship
Tests correlation between economic search interest and market volatility

Author: Simla Tükenmez
Student ID: 32613
Course: DSA210 - Fall 2025-2026

Hypothesis: Higher Google search volume for economic terms (dolar kuru, 
enflasyon, ekonomik kriz) correlates with higher BIST100 volatility

Method: Correlation analysis + Regression
Variables:
- X: Google Trends search volume (composite index)
- Y: BIST100 volatility (rolling std, absolute returns)
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_data():
    """Load master dataset"""
    print("Loading data from master_data_with_sentiment.xlsx...")
    
    try:
        df = pd.read_excel('master_data_with_sentiment.xlsx')
        print("   ✅ Loaded from Excel (with sentiment)")
    except FileNotFoundError:
        try:
            df = pd.read_excel('master_data.xlsx')
            print("   ✅ Loaded from Excel (without sentiment)")
        except FileNotFoundError:
            df = pd.read_csv('master_data.csv')
            print("   ✅ Loaded from CSV")
    
    df['date'] = pd.to_datetime(df['date'])
    print(f"   ✅ {len(df)} rows, {len(df.columns)} columns\n")
    return df


def h4_google_trends_volatility(df):
    """
    Test H4: Google Trends search volume correlates with BIST100 volatility
    
    Measures:
    1. Correlation between search volume and volatility
    2. Regression: Volatility ~ Search Volume
    3. Lead/lag analysis
    """
    
    print("="*80)
    print("H4: GOOGLE TRENDS - VOLATILITY CORRELATION")
    print("="*80)
    
    print(f"\nGoogle Trends Variables Available:")
    trends_cols = [c for c in df.columns if any(x in c.lower() for x in ['dolar', 'enflasyon', 'kriz', 'ekonomik'])]
    for col in trends_cols:
        print(f"   - {col}")
    
    # Create composite panic index from Google Trends
    print(f"\nCreating Composite Digital Panic Index...")
    
    # Select key trends variables
    key_trends = ['dolar_kuru', 'enflasyon', 'ekonomik_kriz']
    
    # Check which are available
    available_trends = [col for col in key_trends if col in df.columns]
    print(f"   Available trends: {available_trends}")
    
    if len(available_trends) == 0:
        print("\nERROR: No Google Trends data found!")
        print("   Please ensure master data has Google Trends columns.")
        return None
    
    # Normalize each trend to 0-1 scale
    for col in available_trends:
        df[f'{col}_normalized'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
    # Create composite index (average of normalized trends)
    normalized_cols = [f'{col}_normalized' for col in available_trends]
    df['Digital_Panic_Index'] = df[normalized_cols].mean(axis=1)
    
    print(f"   ✅ Created Digital_Panic_Index (average of {len(available_trends)} trends)")
    print(f"   Range: {df['Digital_Panic_Index'].min():.4f} to {df['Digital_Panic_Index'].max():.4f}")
    
    # Create volatility measures
    print(f"\nCreating Volatility Measures...")
    
    # 1. Absolute returns (simple volatility proxy)
    df['Abs_Return'] = df['BIST100_Return'].abs()
    
    # 2. Rolling standard deviation (7-day window)
    df['Volatility_7d'] = df['BIST100_Return'].rolling(window=7).std()
    
    # 3. Rolling standard deviation (30-day window)
    df['Volatility_30d'] = df['BIST100_Return'].rolling(window=30).std()
    
    print(f"   ✅ Abs_Return (absolute daily returns)")
    print(f"   ✅ Volatility_7d (7-day rolling std)")
    print(f"   ✅ Volatility_30d (30-day rolling std)")
    
    # Filter data (remove NaNs)
    analysis_df = df[['date', 'Digital_Panic_Index', 'Abs_Return', 
                      'Volatility_7d', 'Volatility_30d', 'BIST100_Return']].dropna()
    
    print(f"\nAnalysis Dataset:")
    print(f"   Total observations: {len(analysis_df)}")
    print(f"   Date range: {analysis_df['date'].min().date()} to {analysis_df['date'].max().date()}")
    
    # Descriptive statistics
    print(f"\n" + "="*80)
    print("DESCRIPTIVE STATISTICS")
    print("="*80)
    
    print(f"\nDigital Panic Index (Google Trends Composite):")
    print(f"   Mean:   {analysis_df['Digital_Panic_Index'].mean():>8.4f}")
    print(f"   Median: {analysis_df['Digital_Panic_Index'].median():>8.4f}")
    print(f"   Std:    {analysis_df['Digital_Panic_Index'].std():>8.4f}")
    print(f"   Min:    {analysis_df['Digital_Panic_Index'].min():>8.4f}")
    print(f"   Max:    {analysis_df['Digital_Panic_Index'].max():>8.4f}")
    
    print(f"\nVolatility Measures:")
    print(f"   Absolute Returns:")
    print(f"     Mean:   {analysis_df['Abs_Return'].mean():>8.4f}%")
    print(f"     Median: {analysis_df['Abs_Return'].median():>8.4f}%")
    
    print(f"\n   7-day Rolling Volatility:")
    print(f"     Mean:   {analysis_df['Volatility_7d'].mean():>8.4f}%")
    print(f"     Median: {analysis_df['Volatility_7d'].median():>8.4f}%")
    
    print(f"\n   30-day Rolling Volatility:")
    print(f"     Mean:   {analysis_df['Volatility_30d'].mean():>8.4f}%")
    print(f"     Median: {analysis_df['Volatility_30d'].median():>8.4f}%")
    
    # Correlation Analysis
    print(f"\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    print(f"\nHypothesis: Higher search interest → Higher volatility")
    
    # Calculate correlations
    corr_abs = analysis_df[['Digital_Panic_Index', 'Abs_Return']].corr().iloc[0, 1]
    corr_7d = analysis_df[['Digital_Panic_Index', 'Volatility_7d']].corr().iloc[0, 1]
    corr_30d = analysis_df[['Digital_Panic_Index', 'Volatility_30d']].corr().iloc[0, 1]
    
    # Statistical significance tests
    n = len(analysis_df)
    
    # Absolute returns
    t_stat_abs = corr_abs * np.sqrt(n - 2) / np.sqrt(1 - corr_abs**2)
    p_value_abs = 2 * (1 - stats.t.cdf(abs(t_stat_abs), n - 2))
    
    # 7-day volatility
    t_stat_7d = corr_7d * np.sqrt(n - 2) / np.sqrt(1 - corr_7d**2)
    p_value_7d = 2 * (1 - stats.t.cdf(abs(t_stat_7d), n - 2))
    
    # 30-day volatility
    t_stat_30d = corr_30d * np.sqrt(n - 2) / np.sqrt(1 - corr_30d**2)
    p_value_30d = 2 * (1 - stats.t.cdf(abs(t_stat_30d), n - 2))
    
    print(f"\nCorrelation Results:")
    print(f"\n   Digital Panic Index ↔ Absolute Returns:")
    print(f"     r = {corr_abs:>8.4f}")
    print(f"     p = {p_value_abs:>8.4f} {'***' if p_value_abs < 0.01 else '**' if p_value_abs < 0.05 else '*' if p_value_abs < 0.10 else ''}")
    
    print(f"\n   Digital Panic Index ↔ Volatility (7-day):")
    print(f"     r = {corr_7d:>8.4f}")
    print(f"     p = {p_value_7d:>8.4f} {'***' if p_value_7d < 0.01 else '**' if p_value_7d < 0.05 else '*' if p_value_7d < 0.10 else ''}")
    
    print(f"\n   Digital Panic Index ↔ Volatility (30-day):")
    print(f"     r = {corr_30d:>8.4f}")
    print(f"     p = {p_value_30d:>8.4f} {'***' if p_value_30d < 0.01 else '**' if p_value_30d < 0.05 else '*' if p_value_30d < 0.10 else ''}")
    
    # Interpretation
    print(f"\nInterpretation:")
    
    # Use 7-day volatility as primary measure
    primary_corr = corr_7d
    primary_p = p_value_7d
    
    if primary_p < 0.05:
        print(f"   ✅ SIGNIFICANT positive correlation found")
        if primary_corr > 0.3:
            print(f"   → MODERATE relationship (r > 0.3)")
        elif primary_corr > 0.1:
            print(f"   → WEAK relationship (0.1 < r < 0.3)")
        else:
            print(f"   → VERY WEAK relationship (r < 0.1)")
        print(f"   → Higher search interest DOES predict higher volatility")
    else:
        print(f"   ❌ No significant correlation found")
        print(f"   → Search interest does NOT reliably predict volatility")
    
    # Regression Analysis
    print(f"\n" + "="*80)
    print("REGRESSION ANALYSIS")
    print("="*80)
    
    print(f"\nModel: Volatility_7d = β₀ + β₁ * Digital_Panic_Index + ε")
    
    X = analysis_df[['Digital_Panic_Index']].values
    y = analysis_df['Volatility_7d'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    beta0 = model.intercept_
    beta1 = model.coef_[0]
    r2 = model.score(X, y)
    
    # Calculate p-value for beta1
    y_pred = model.predict(X)
    residuals = y - y_pred
    mse = np.sum(residuals**2) / (len(y) - 2)
    se_beta1 = np.sqrt(mse / np.sum((X - X.mean())**2))
    t_stat = beta1 / se_beta1
    p_value_beta1 = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - 2))
    
    print(f"\nRegression Results:")
    print(f"   β₀ (Intercept):           {beta0:>8.4f}")
    print(f"   β₁ (Panic Index):         {beta1:>8.4f} {'***' if p_value_beta1 < 0.01 else '**' if p_value_beta1 < 0.05 else '*' if p_value_beta1 < 0.10 else ''}")
    print(f"   SE(β₁):                   {se_beta1:>8.4f}")
    print(f"   t-statistic:              {t_stat:>8.4f}")
    print(f"   p-value:                  {p_value_beta1:>8.4f}")
    print(f"   R²:                       {r2:>8.4f}")
    print(f"   N:                        {len(y)}")
    
    print(f"\nStatistical Decision at α = 0.05:")
    if p_value_beta1 < 0.05:
        print(f"   ✅ β₁ is SIGNIFICANT")
        print(f"   → Digital panic index predicts volatility")
        print(f"   → 1-unit increase in panic → +{beta1:.4f}% volatility")
    else:
        print(f"   ❌ β₁ is NOT significant")
        print(f"   → Panic index does not reliably predict volatility")
    
    # Individual trends analysis
    print(f"\n" + "="*80)
    print("INDIVIDUAL TRENDS ANALYSIS")
    print("="*80)
    
    print(f"\nCorrelations with Volatility (7-day):")
    for trend in available_trends:
        if trend in df.columns:
            trend_data = df[[trend, 'Volatility_7d']].dropna()
            if len(trend_data) > 0:
                corr = trend_data.corr().iloc[0, 1]
                n_trend = len(trend_data)
                t_stat_trend = corr * np.sqrt(n_trend - 2) / np.sqrt(1 - corr**2)
                p_trend = 2 * (1 - stats.t.cdf(abs(t_stat_trend), n_trend - 2))
                print(f"   {trend:20s}: r = {corr:>7.4f}, p = {p_trend:>7.4f} {'***' if p_trend < 0.01 else '**' if p_trend < 0.05 else '*' if p_trend < 0.10 else ''}")
    
    # Event analysis - major spikes
    print(f"\n" + "="*80)
    print("HIGH PANIC EPISODES")
    print("="*80)
    
    # Top 10 highest panic days
    top_panic = analysis_df.nlargest(10, 'Digital_Panic_Index')[['date', 'Digital_Panic_Index', 'Volatility_7d', 'BIST100_Return']]
    
    print(f"\nTop 10 Highest Digital Panic Days:")
    print(f"{'Date':<12} {'Panic Index':<12} {'Volatility':<12} {'BIST Return':<12}")
    print("-" * 50)
    for idx, row in top_panic.iterrows():
        print(f"{row['date'].strftime('%Y-%m-%d'):<12} {row['Digital_Panic_Index']:>11.4f} {row['Volatility_7d']:>11.4f}% {row['BIST100_Return']:>11.2f}%")
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    visualize_trends_volatility(analysis_df, df, available_trends,
                                corr_7d, p_value_7d, beta1, r2)
    
    # Save results
    save_h4_results(analysis_df, available_trends, 
                   corr_abs, p_value_abs,
                   corr_7d, p_value_7d,
                   corr_30d, p_value_30d,
                   beta0, beta1, r2, p_value_beta1)
    
    return {
        'correlation': corr_7d,
        'p_value_corr': p_value_7d,
        'beta1': beta1,
        'p_value_beta1': p_value_beta1,
        'r2': r2,
        'significant': p_value_7d < 0.05
    }


def visualize_trends_volatility(analysis_df, full_df, trends_list, 
                                corr, p_value, beta1, r2):
    """Create comprehensive visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    # Plot 1: Time series - Panic index and volatility
    ax1 = axes[0, 0]
    
    ax1_twin = ax1.twinx()
    
    # Panic index (left axis)
    ax1.plot(analysis_df['date'], analysis_df['Digital_Panic_Index'], 
            linewidth=2, color='#E63946', label='Digital Panic Index', alpha=0.8)
    ax1.fill_between(analysis_df['date'], 0, analysis_df['Digital_Panic_Index'],
                     alpha=0.2, color='#E63946')
    
    # Volatility (right axis)
    ax1_twin.plot(analysis_df['date'], analysis_df['Volatility_7d'],
                 linewidth=2, color='#2E86AB', label='7-day Volatility', alpha=0.8, linestyle='--')
    
    ax1.set_title('Digital Panic Index vs BIST100 Volatility Over Time',
                 fontsize=13, fontweight='bold', pad=15)
    ax1.set_ylabel('Digital Panic Index (0-1)', fontsize=11, color='#E63946')
    ax1_twin.set_ylabel('Volatility (%)', fontsize=11, color='#2E86AB')
    ax1.tick_params(axis='y', labelcolor='#E63946')
    ax1_twin.tick_params(axis='y', labelcolor='#2E86AB')
    ax1.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    # Plot 2: Scatter plot with regression line
    ax2 = axes[0, 1]
    
    ax2.scatter(analysis_df['Digital_Panic_Index'], analysis_df['Volatility_7d'],
               alpha=0.4, s=30, color='#457B9D', edgecolors='white', linewidth=0.3)
    
    # Regression line
    X_range = np.linspace(analysis_df['Digital_Panic_Index'].min(),
                         analysis_df['Digital_Panic_Index'].max(), 100)
    y_range = beta1 * X_range + (analysis_df['Volatility_7d'].mean() - beta1 * analysis_df['Digital_Panic_Index'].mean())
    ax2.plot(X_range, y_range, 'r-', linewidth=2.5, label=f'Regression: β₁={beta1:.3f}')
    
    ax2.set_title('Digital Panic Index vs Volatility\n(7-day Rolling)',
                 fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlabel('Digital Panic Index', fontsize=11)
    ax2.set_ylabel('Volatility (%)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Add stats box
    stats_text = f'r = {corr:.3f}\n'
    stats_text += f'p = {p_value:.4f}\n'
    stats_text += f'R² = {r2:.3f}'
    ax2.text(0.05, 0.95, stats_text,
            transform=ax2.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow' if p_value < 0.05 else 'lightgray', alpha=0.8))
    
    # Plot 3: Individual trends comparison
    ax3 = axes[1, 0]
    
    # Calculate correlation for each trend
    trend_corrs = []
    trend_names = []
    for trend in trends_list:
        if trend in full_df.columns:
            temp_df = full_df[[trend, 'Volatility_7d']].dropna()
            if len(temp_df) > 0:
                corr_val = temp_df.corr().iloc[0, 1]
                trend_corrs.append(corr_val)
                # Clean name for display
                clean_name = trend.replace('_', ' ').title()
                trend_names.append(clean_name)
    
    # Add composite
    trend_names.append('Composite')
    trend_corrs.append(corr)
    
    colors = ['#2E86AB'] * len(trends_list) + ['#E63946']
    bars = ax3.barh(trend_names, trend_corrs, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax3.axvline(0, color='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, trend_corrs):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}',
                ha='left' if width > 0 else 'right',
                va='center', fontsize=9, fontweight='bold')
    
    ax3.set_title('Correlation with Volatility: Individual vs Composite',
                 fontsize=13, fontweight='bold', pad=15)
    ax3.set_xlabel('Correlation Coefficient', fontsize=11)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Distribution comparison - High vs Low panic
    ax4 = axes[1, 1]
    
    # Split by median panic index
    median_panic = analysis_df['Digital_Panic_Index'].median()
    high_panic = analysis_df[analysis_df['Digital_Panic_Index'] > median_panic]['Volatility_7d']
    low_panic = analysis_df[analysis_df['Digital_Panic_Index'] <= median_panic]['Volatility_7d']
    
    ax4.hist(low_panic, bins=30, alpha=0.6, color='#2E86AB',
            label=f'Low Panic (σ={low_panic.std():.2f}%)', edgecolor='black', linewidth=0.5)
    ax4.hist(high_panic, bins=30, alpha=0.6, color='#E63946',
            label=f'High Panic (σ={high_panic.std():.2f}%)', edgecolor='black', linewidth=0.5)
    
    ax4.axvline(low_panic.mean(), color='#2E86AB', linestyle='--', linewidth=2, label=f'Low mean: {low_panic.mean():.2f}%')
    ax4.axvline(high_panic.mean(), color='#E63946', linestyle='--', linewidth=2, label=f'High mean: {high_panic.mean():.2f}%')
    
    ax4.set_title('Volatility Distribution: High vs Low Panic Periods',
                 fontsize=13, fontweight='bold', pad=15)
    ax4.set_xlabel('Volatility (%)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('h4_google_trends_volatility.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: h4_google_trends_volatility.png")
    plt.close()


def save_h4_results(analysis_df, trends_list,
                   corr_abs, p_abs, corr_7d, p_7d, corr_30d, p_30d,
                   beta0, beta1, r2, p_beta1):
    """Save H4 results"""
    
    output_path = 'h4_results.txt'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("H4: GOOGLE TRENDS - VOLATILITY CORRELATION - RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Student: Simla Tükenmez (32613)\n")
        f.write(f"Course: DSA210 - Fall 2025-2026\n\n")
        
        f.write("="*80 + "\n")
        f.write("RESEARCH QUESTION\n")
        f.write("="*80 + "\n\n")
        f.write("Does Google search interest in economic terms correlate with\n")
        f.write("BIST100 market volatility?\n\n")
        
        f.write("="*80 + "\n")
        f.write("METHODOLOGY\n")
        f.write("="*80 + "\n\n")
        f.write("Digital Panic Index:\n")
        f.write(f"  Composite of {len(trends_list)} Google Trends keywords:\n")
        for trend in trends_list:
            f.write(f"    - {trend}\n")
        f.write("\n  Each normalized to 0-1 scale, then averaged\n\n")
        
        f.write("Volatility Measures:\n")
        f.write("  - Absolute Returns: |daily return|\n")
        f.write("  - 7-day Rolling Volatility: std(returns, 7 days)\n")
        f.write("  - 30-day Rolling Volatility: std(returns, 30 days)\n\n")
        
        f.write(f"Sample Size: {len(analysis_df)} observations\n\n")
        
        f.write("="*80 + "\n")
        f.write("CORRELATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Digital Panic Index ↔ Absolute Returns:\n")
        f.write(f"  r = {corr_abs:>8.4f}\n")
        f.write(f"  p = {p_abs:>8.4f} {'***' if p_abs < 0.01 else '**' if p_abs < 0.05 else '*' if p_abs < 0.10 else ''}\n\n")
        
        f.write(f"Digital Panic Index ↔ Volatility (7-day):\n")
        f.write(f"  r = {corr_7d:>8.4f}\n")
        f.write(f"  p = {p_7d:>8.4f} {'***' if p_7d < 0.01 else '**' if p_7d < 0.05 else '*' if p_7d < 0.10 else ''}\n\n")
        
        f.write(f"Digital Panic Index ↔ Volatility (30-day):\n")
        f.write(f"  r = {corr_30d:>8.4f}\n")
        f.write(f"  p = {p_30d:>8.4f} {'***' if p_30d < 0.01 else '**' if p_30d < 0.05 else '*' if p_30d < 0.10 else ''}\n\n")
        
        f.write("="*80 + "\n")
        f.write("REGRESSION ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write("Model: Volatility_7d = β₀ + β₁ * Digital_Panic_Index + ε\n\n")
        f.write(f"Results:\n")
        f.write(f"  β₀ (Intercept):     {beta0:>10.4f}\n")
        f.write(f"  β₁ (Panic Index):   {beta1:>10.4f} {'***' if p_beta1 < 0.01 else '**' if p_beta1 < 0.05 else '*' if p_beta1 < 0.10 else ''}\n")
        f.write(f"  P-value(β₁):        {p_beta1:>10.4f}\n")
        f.write(f"  R²:                 {r2:>10.4f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("CONCLUSION\n")
        f.write("="*80 + "\n\n")
        
        if p_7d < 0.05:
            f.write("Decision: H4 is SUPPORTED\n\n")
            f.write(f"A SIGNIFICANT positive correlation exists between digital panic\n")
            f.write(f"(Google search interest) and market volatility (r = {corr_7d:.4f}, p = {p_7d:.4f}).\n\n")
            if corr_7d > 0.3:
                f.write("The moderate correlation suggests that increased public anxiety\n")
                f.write("(reflected in Google searches) is associated with higher market volatility.\n")
            else:
                f.write("The weak-to-moderate correlation suggests digital panic has measurable\n")
                f.write("but limited predictive power for market volatility.\n")
        else:
            f.write("Decision: H4 is NOT SUPPORTED\n\n")
            f.write(f"No significant correlation found between digital panic and\n")
            f.write(f"market volatility (r = {corr_7d:.4f}, p = {p_7d:.4f}).\n\n")
            f.write("This suggests Google search behavior does not reliably predict\n")
            f.write("or coincide with market volatility in the Turkish context.\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"   ✅ Saved: {output_path}")


if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("H4: GOOGLE TRENDS - VOLATILITY CORRELATION")
    print("="*80)
    print("\nDSA210 Term Project: Digital Panic and Market Dynamics")
    print("Student: Simla Tükenmez (32613)\n")
    
    # Load data
    df = load_data()
    
    # Run H4 test
    results = h4_google_trends_volatility(df)
    
    if results:
        print("\n" + "="*80)
        print("✅ H4 ANALYSIS COMPLETE!")
        print("="*80)
        
        print("\nOutput Files:")
        print("   1. h4_google_trends_volatility.png - 4-panel visualization")
        print("   2. h4_results.txt - Detailed results")
        
        print("\nQuick Summary:")
        print(f"   Correlation (r):    {results['correlation']:>8.4f}")
        print(f"   P-value:            {results['p_value_corr']:>8.4f}")
        print(f"   Regression β₁:      {results['beta1']:>8.4f}")
        print(f"   R²:                 {results['r2']:>8.4f}")
        print(f"   Result:             {'✅ SIGNIFICANT' if results['significant'] else '❌ NOT SIGNIFICANT'}")
        
        if results['significant']:
            print("\nCONCLUSION:")
            print("   H4 is SUPPORTED by the data.")
            print("   Digital panic (Google searches) correlates with volatility.")
        else:
            print("\nCONCLUSION:")
            print("   H4 is NOT supported.")
            print("   No significant correlation found.")
