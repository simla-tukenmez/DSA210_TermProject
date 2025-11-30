"""
H3: Structural Break Analysis
Tests whether June 2023 policy change created structural break in 
BIST100-funding cost relationship

Author: Simla Tükenmez
Student ID: 32613
Course: DSA210 - Fall 2025-2026

Event: June 23, 2023 - New economic management team
       Interest rates: 8.5% → 50% (by March 2024)

Hypothesis: The relationship between BIST100 returns and funding cost 
changes differs significantly before vs after June 23, 2023

Method: 
1. Split-sample regression (Pre vs Post)
2. Chow test for structural break
3. Interaction term test
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


def h3_structural_break_test(df):
    """
    Test H3: Structural break in BIST100-funding cost relationship
    
    Split date: June 23, 2023 (new economic team appointed)
    
    Model: BIST100_Return = β0 + β1 * Funding_Cost_Change + ε
    
    Test: β1_pre ≠ β1_post (different sensitivities to rate changes)
    """
    
    print("="*80)
    print("H3: STRUCTURAL BREAK ANALYSIS")
    print("="*80)
    
    # Define break date
    break_date = pd.Timestamp('2023-06-23')
    
    print(f"\nStructural Break Date:")
    print(f"   June 23, 2023 - New economic management appointed")
    print(f"   → Mehmet Şimşek (Finance Minister)")
    print(f"   → Hafize Gaye Erkan (TCMB Governor)")
    print(f"   → Policy shift: Unorthodox → Orthodox monetary policy")
    
    # Split data
    pre_period = df[df['date'] < break_date].copy()
    post_period = df[df['date'] >= break_date].copy()
    
    print(f"\nSample Split:")
    print(f"   Pre-period:  {pre_period['date'].min().date()} to {pre_period['date'].max().date()}")
    print(f"   Post-period: {post_period['date'].min().date()} to {post_period['date'].max().date()}")
    print(f"\n   Pre-period observations:  {len(pre_period)}")
    print(f"   Post-period observations: {len(post_period)}")
    
    # Prepare regression data (drop NaNs)
    pre_data = pre_period[['BIST100_Return', 'Funding_Cost_Change']].dropna()
    post_data = post_period[['BIST100_Return', 'Funding_Cost_Change']].dropna()
    
    print(f"\n   Usable observations (with both variables):")
    print(f"   Pre:  {len(pre_data)}")
    print(f"   Post: {len(post_data)}")
    
    # Check if we have enough data
    if len(pre_data) < 30 or len(post_data) < 30:
        print(f"\nWARNING: Small sample size may affect results")
    
    # Descriptive statistics
    print(f"\n" + "="*80)
    print("DESCRIPTIVE STATISTICS")
    print("="*80)
    
    print(f"\nPre-Period (Jan 2023 - Jun 22, 2023):")
    print(f"   Mean BIST return:       {pre_data['BIST100_Return'].mean():>8.4f}%")
    print(f"   Mean funding cost Δ:    {pre_data['Funding_Cost_Change'].mean():>8.4f} pp")
    print(f"   Funding cost range:     {pre_data['Funding_Cost_Change'].min():.2f} to {pre_data['Funding_Cost_Change'].max():.2f} pp")
    print(f"   Correlation:            {pre_data.corr().iloc[0,1]:>8.4f}")
    
    print(f"\nPost-Period (Jun 23, 2023 onwards):")
    print(f"   Mean BIST return:       {post_data['BIST100_Return'].mean():>8.4f}%")
    print(f"   Mean funding cost Δ:    {post_data['Funding_Cost_Change'].mean():>8.4f} pp")
    print(f"   Funding cost range:     {post_data['Funding_Cost_Change'].min():.2f} to {post_data['Funding_Cost_Change'].max():.2f} pp")
    print(f"   Correlation:            {post_data.corr().iloc[0,1]:>8.4f}")
    
    # Policy context
    print(f"\nPolicy Context:")
    pre_avg_rate = pre_period['Funding_Cost'].mean()
    post_avg_rate = post_period['Funding_Cost'].mean()
    print(f"   Pre-period avg funding cost:  {pre_avg_rate:.2f}%")
    print(f"   Post-period avg funding cost: {post_avg_rate:.2f}%")
    print(f"   Change: +{post_avg_rate - pre_avg_rate:.2f} percentage points")
    
    # Run split-sample regressions
    print(f"\n" + "="*80)
    print("SPLIT-SAMPLE REGRESSION ANALYSIS")
    print("="*80)
    
    print(f"\nModel: BIST100_Return = β0 + β1 * Funding_Cost_Change + ε")
    
    # Pre-period regression
    X_pre = pre_data[['Funding_Cost_Change']].values
    y_pre = pre_data['BIST100_Return'].values
    
    model_pre = LinearRegression()
    model_pre.fit(X_pre, y_pre)
    
    beta0_pre = model_pre.intercept_
    beta1_pre = model_pre.coef_[0]
    r2_pre = model_pre.score(X_pre, y_pre)
    
    # Calculate t-statistic and p-value for beta1_pre
    y_pred_pre = model_pre.predict(X_pre)
    residuals_pre = y_pre - y_pred_pre
    mse_pre = np.sum(residuals_pre**2) / (len(y_pre) - 2)
    se_beta1_pre = np.sqrt(mse_pre / np.sum((X_pre - X_pre.mean())**2))
    t_stat_pre = beta1_pre / se_beta1_pre
    p_value_pre = 2 * (1 - stats.t.cdf(abs(t_stat_pre), len(y_pre) - 2))
    
    print(f"\nPRE-PERIOD REGRESSION (Jan - Jun 22, 2023):")
    print(f"   β0 (Intercept):           {beta0_pre:>8.4f}")
    print(f"   β1 (Funding Cost Δ):      {beta1_pre:>8.4f}")
    print(f"   SE(β1):                   {se_beta1_pre:>8.4f}")
    print(f"   t-statistic:              {t_stat_pre:>8.4f}")
    print(f"   p-value:                  {p_value_pre:>8.4f} {'***' if p_value_pre < 0.01 else '**' if p_value_pre < 0.05 else '*' if p_value_pre < 0.10 else ''}")
    print(f"   R²:                       {r2_pre:>8.4f}")
    print(f"   N:                        {len(y_pre)}")
    
    # Post-period regression
    X_post = post_data[['Funding_Cost_Change']].values
    y_post = post_data['BIST100_Return'].values
    
    model_post = LinearRegression()
    model_post.fit(X_post, y_post)
    
    beta0_post = model_post.intercept_
    beta1_post = model_post.coef_[0]
    r2_post = model_post.score(X_post, y_post)
    
    # Calculate t-statistic and p-value for beta1_post
    y_pred_post = model_post.predict(X_post)
    residuals_post = y_post - y_pred_post
    mse_post = np.sum(residuals_post**2) / (len(y_post) - 2)
    se_beta1_post = np.sqrt(mse_post / np.sum((X_post - X_post.mean())**2))
    t_stat_post = beta1_post / se_beta1_post
    p_value_post = 2 * (1 - stats.t.cdf(abs(t_stat_post), len(y_post) - 2))
    
    print(f"\nPOST-PERIOD REGRESSION (Jun 23, 2023 onwards):")
    print(f"   β0 (Intercept):           {beta0_post:>8.4f}")
    print(f"   β1 (Funding Cost Δ):      {beta1_post:>8.4f}")
    print(f"   SE(β1):                   {se_beta1_post:>8.4f}")
    print(f"   t-statistic:              {t_stat_post:>8.4f}")
    print(f"   p-value:                  {p_value_post:>8.4f} {'***' if p_value_post < 0.01 else '**' if p_value_post < 0.05 else '*' if p_value_post < 0.10 else ''}")
    print(f"   R²:                       {r2_post:>8.4f}")
    print(f"   N:                        {len(y_post)}")
    
    # Compare coefficients
    print(f"\nCOEFFICIENT COMPARISON:")
    print(f"   Δβ1 (Post - Pre):         {beta1_post - beta1_pre:>8.4f}")
    print(f"   Relative change:          {((beta1_post - beta1_pre) / abs(beta1_pre) * 100) if beta1_pre != 0 else float('inf'):>8.1f}%")
    
    if abs(beta1_post) > abs(beta1_pre):
        print(f"   → Post-period shows STRONGER sensitivity to rate changes")
    else:
        print(f"   → Pre-period shows STRONGER sensitivity to rate changes")
    
    # Chow Test for Structural Break
    print(f"\n" + "="*80)
    print("CHOW TEST FOR STRUCTURAL BREAK")
    print("="*80)
    
    print(f"\nHypotheses:")
    print(f"   H0: No structural break (β1_pre = β1_post)")
    print(f"   H3: Structural break exists (β1_pre ≠ β1_post)")
    
    # Pooled regression (full sample)
    full_data = pd.concat([pre_data, post_data])
    X_full = full_data[['Funding_Cost_Change']].values
    y_full = full_data['BIST100_Return'].values
    
    model_full = LinearRegression()
    model_full.fit(X_full, y_full)
    
    # Calculate RSS
    rss_full = np.sum((y_full - model_full.predict(X_full))**2)
    rss_pre = np.sum(residuals_pre**2)
    rss_post = np.sum(residuals_post**2)
    rss_split = rss_pre + rss_post
    
    # Chow test statistic
    k = 2  # number of parameters (intercept + slope)
    n1 = len(y_pre)
    n2 = len(y_post)
    
    chow_stat = ((rss_full - rss_split) / k) / (rss_split / (n1 + n2 - 2*k))
    p_value_chow = 1 - stats.f.cdf(chow_stat, k, n1 + n2 - 2*k)
    
    print(f"\nChow Test Results:")
    print(f"   RSS (pooled):             {rss_full:>12.4f}")
    print(f"   RSS (split):              {rss_split:>12.4f}")
    print(f"   Chow F-statistic:         {chow_stat:>12.4f}")
    print(f"   Degrees of freedom:       ({k}, {n1 + n2 - 2*k})")
    print(f"   P-value:                  {p_value_chow:>12.4f}")
    
    print(f"\nStatistical Decision at α = 0.05:")
    if p_value_chow < 0.05:
        print(f"   ✅ REJECT H0")
        print(f"   → STRUCTURAL BREAK confirmed")
        print(f"   → Relationship changed significantly after Jun 23, 2023")
    else:
        print(f"   ❌ FAIL TO REJECT H0")
        print(f"   → NO significant structural break")
        print(f"   → Relationship remained stable across periods")
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    visualize_structural_break(df, pre_data, post_data, 
                               model_pre, model_post, break_date,
                               beta1_pre, beta1_post, p_value_chow)
    
    # Save results
    save_h3_results(pre_data, post_data, 
                   beta0_pre, beta1_pre, r2_pre, p_value_pre,
                   beta0_post, beta1_post, r2_post, p_value_post,
                   chow_stat, p_value_chow)
    
    return {
        'beta1_pre': beta1_pre,
        'beta1_post': beta1_post,
        'p_value_pre': p_value_pre,
        'p_value_post': p_value_post,
        'chow_stat': chow_stat,
        'p_value_chow': p_value_chow,
        'structural_break': p_value_chow < 0.05
    }


def visualize_structural_break(df, pre_data, post_data, 
                               model_pre, model_post, break_date,
                               beta1_pre, beta1_post, p_value_chow):
    """Create comprehensive visualizations for structural break"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    # Plot 1: Funding cost over time with break point
    ax1 = axes[0, 0]
    
    data_2023_2024 = df[(df['date'] >= '2023-01-01') & (df['date'] <= '2024-12-31')]
    
    ax1.plot(data_2023_2024['date'], data_2023_2024['Funding_Cost'], 
            linewidth=2.5, color='#E63946', label='Funding Cost')
    ax1.axvline(break_date, color='black', linestyle='--', linewidth=2.5, 
               label='Policy Change (Jun 23)', alpha=0.8)
    
    # Shade regimes
    ax1.axvspan(data_2023_2024['date'].min(), break_date, 
                alpha=0.15, color='blue', label='Pre-Tightening')
    ax1.axvspan(break_date, data_2023_2024['date'].max(), 
                alpha=0.15, color='red', label='Tightening')
    
    ax1.set_title('TCMB Funding Cost: Pre vs Post Policy Change', 
                 fontsize=13, fontweight='bold', pad=15)
    ax1.set_ylabel('Funding Cost (%)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plots with regression lines
    ax2 = axes[0, 1]
    
    # Pre-period scatter
    ax2.scatter(pre_data['Funding_Cost_Change'], pre_data['BIST100_Return'],
               alpha=0.5, s=40, color='#2E86AB', label='Pre-Period', edgecolors='white', linewidth=0.5)
    
    # Pre-period regression line
    X_range_pre = np.linspace(pre_data['Funding_Cost_Change'].min(), 
                              pre_data['Funding_Cost_Change'].max(), 100).reshape(-1, 1)
    y_pred_range_pre = model_pre.predict(X_range_pre)
    ax2.plot(X_range_pre, y_pred_range_pre, color='#2E86AB', linewidth=2.5, 
            label=f'Pre: β₁={beta1_pre:.3f}', linestyle='-')
    
    # Post-period scatter
    ax2.scatter(post_data['Funding_Cost_Change'], post_data['BIST100_Return'],
               alpha=0.5, s=40, color='#E63946', label='Post-Period', edgecolors='white', linewidth=0.5)
    
    # Post-period regression line
    X_range_post = np.linspace(post_data['Funding_Cost_Change'].min(), 
                               post_data['Funding_Cost_Change'].max(), 100).reshape(-1, 1)
    y_pred_range_post = model_post.predict(X_range_post)
    ax2.plot(X_range_post, y_pred_range_post, color='#E63946', linewidth=2.5, 
            label=f'Post: β₁={beta1_post:.3f}', linestyle='-')
    
    ax2.axhline(0, color='black', linestyle=':', linewidth=1)
    ax2.axvline(0, color='black', linestyle=':', linewidth=1)
    
    ax2.set_title('BIST100 Return vs Funding Cost Change\n(Pre vs Post Policy Change)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlabel('Funding Cost Change (pp)', fontsize=11)
    ax2.set_ylabel('BIST100 Return (%)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Add Chow test result
    chow_text = f'Chow test p={p_value_chow:.4f}\n'
    chow_text += 'Structural break: ' + ('YES ✓' if p_value_chow < 0.05 else 'NO')
    ax2.text(0.05, 0.95, chow_text,
            transform=ax2.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow' if p_value_chow < 0.05 else 'lightgray', alpha=0.8))
    
    # Plot 3: Coefficient comparison
    ax3 = axes[1, 0]
    
    periods = ['Pre-Period\n(Jan-Jun 22)', 'Post-Period\n(Jun 23+)']
    coefficients = [beta1_pre, beta1_post]
    colors_bar = ['#2E86AB', '#E63946']
    
    bars = ax3.bar(periods, coefficients, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.axhline(0, color='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, coef in zip(bars, coefficients):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{coef:.4f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=11, fontweight='bold')
    
    ax3.set_title('Funding Cost Sensitivity Coefficient (β₁) Comparison', 
                 fontsize=13, fontweight='bold', pad=15)
    ax3.set_ylabel('β₁ (Sensitivity to Rate Changes)', fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Rolling correlation
    ax4 = axes[1, 1]
    
    # Calculate 90-day rolling correlation
    window = 90
    df_sorted = df.sort_values('date').copy()
    df_sorted['Rolling_Corr'] = df_sorted['BIST100_Return'].rolling(window).corr(
        df_sorted['Funding_Cost_Change'])
    
    plot_data = df_sorted[(df_sorted['date'] >= '2023-01-01') & 
                          (df_sorted['date'] <= '2024-12-31')]
    
    ax4.plot(plot_data['date'], plot_data['Rolling_Corr'], 
            linewidth=2, color='#2E86AB', label=f'{window}-day Rolling Correlation')
    ax4.axhline(0, color='black', linestyle=':', linewidth=1)
    ax4.axvline(break_date, color='red', linestyle='--', linewidth=2.5, 
               label='Policy Change', alpha=0.8)
    
    ax4.set_title(f'{window}-Day Rolling Correlation:\nBIST100 Return vs Funding Cost Change', 
                 fontsize=13, fontweight='bold', pad=15)
    ax4.set_ylabel('Correlation Coefficient', fontsize=11)
    ax4.set_xlabel('Date', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.savefig('h3_structural_break.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: h3_structural_break.png")
    plt.close()


def save_h3_results(pre_data, post_data,
                   beta0_pre, beta1_pre, r2_pre, p_value_pre,
                   beta0_post, beta1_post, r2_post, p_value_post,
                   chow_stat, p_value_chow):
    """Save H3 test results"""
    
    output_path = 'h3_results.txt'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("H3: STRUCTURAL BREAK ANALYSIS - RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Student: Simla Tükenmez (32613)\n")
        f.write(f"Course: DSA210 - Fall 2025-2026\n\n")
        
        f.write("="*80 + "\n")
        f.write("RESEARCH QUESTION\n")
        f.write("="*80 + "\n\n")
        f.write("Did the June 23, 2023 economic policy change create a structural break\n")
        f.write("in the relationship between BIST100 returns and TCMB funding cost changes?\n\n")
        
        f.write("="*80 + "\n")
        f.write("EVENT CONTEXT\n")
        f.write("="*80 + "\n\n")
        f.write("Date: June 23, 2023\n")
        f.write("Event: New economic management team appointed\n")
        f.write("  - Mehmet Şimşek: Finance Minister\n")
        f.write("  - Hafize Gaye Erkan: TCMB Governor\n\n")
        f.write("Policy Shift: Unorthodox → Orthodox monetary policy\n")
        f.write("  - Interest rates: 8.5% → 50% (by March 2024)\n")
        f.write("  - Return to inflation targeting\n\n")
        
        f.write("="*80 + "\n")
        f.write("REGRESSION MODEL\n")
        f.write("="*80 + "\n\n")
        f.write("BIST100_Return = β₀ + β₁ * Funding_Cost_Change + ε\n\n")
        f.write("Where:\n")
        f.write("  BIST100_Return: Daily log return (%)\n")
        f.write("  Funding_Cost_Change: Daily change in funding cost (pp)\n\n")
        
        f.write("="*80 + "\n")
        f.write("PRE-PERIOD REGRESSION (Jan 2023 - Jun 22, 2023)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Observations: {len(pre_data)}\n\n")
        f.write(f"Coefficients:\n")
        f.write(f"  β₀ (Intercept):           {beta0_pre:>10.4f}\n")
        f.write(f"  β₁ (Funding Cost Δ):      {beta1_pre:>10.4f} {'***' if p_value_pre < 0.01 else '**' if p_value_pre < 0.05 else '*' if p_value_pre < 0.10 else ''}\n")
        f.write(f"  P-value(β₁):              {p_value_pre:>10.4f}\n")
        f.write(f"  R²:                       {r2_pre:>10.4f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("POST-PERIOD REGRESSION (Jun 23, 2023 onwards)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Observations: {len(post_data)}\n\n")
        f.write(f"Coefficients:\n")
        f.write(f"  β₀ (Intercept):           {beta0_post:>10.4f}\n")
        f.write(f"  β₁ (Funding Cost Δ):      {beta1_post:>10.4f} {'***' if p_value_post < 0.01 else '**' if p_value_post < 0.05 else '*' if p_value_post < 0.10 else ''}\n")
        f.write(f"  P-value(β₁):              {p_value_post:>10.4f}\n")
        f.write(f"  R²:                       {r2_post:>10.4f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("CHOW TEST FOR STRUCTURAL BREAK\n")
        f.write("="*80 + "\n\n")
        f.write("Hypotheses:\n")
        f.write("  H₀: β₁_pre = β₁_post (no structural break)\n")
        f.write("  H₃: β₁_pre ≠ β₁_post (structural break exists)\n\n")
        f.write(f"Test Results:\n")
        f.write(f"  Chow F-statistic:         {chow_stat:>10.4f}\n")
        f.write(f"  P-value:                  {p_value_chow:>10.4f}\n")
        f.write(f"  Significance level:       α = 0.05\n\n")
        
        f.write("="*80 + "\n")
        f.write("CONCLUSION\n")
        f.write("="*80 + "\n\n")
        
        if p_value_chow < 0.05:
            f.write("Decision: REJECT H₀ at α = 0.05\n\n")
            f.write("H3 is SUPPORTED. A STRUCTURAL BREAK is confirmed.\n\n")
            f.write(f"The sensitivity of BIST100 returns to funding cost changes\n")
            f.write(f"changed significantly after the June 23, 2023 policy shift:\n\n")
            f.write(f"  Pre-period:  β₁ = {beta1_pre:.4f}\n")
            f.write(f"  Post-period: β₁ = {beta1_post:.4f}\n")
            f.write(f"  Change:      Δβ₁ = {beta1_post - beta1_pre:.4f}\n\n")
        else:
            f.write("Decision: FAIL TO REJECT H₀ at α = 0.05\n\n")
            f.write("H3 is NOT SUPPORTED. No significant structural break detected.\n\n")
            f.write(f"Although the coefficients differ:\n")
            f.write(f"  Pre-period:  β₁ = {beta1_pre:.4f}\n")
            f.write(f"  Post-period: β₁ = {beta1_post:.4f}\n\n")
            f.write(f"This difference is not statistically significant (p = {p_value_chow:.4f}).\n\n")
        
        f.write("="*80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*80 + "\n\n")
        
        if p_value_chow < 0.05:
            f.write("The structural break confirms that the June 2023 policy change\n")
            f.write("fundamentally altered how markets respond to monetary policy.\n\n")
            f.write("This likely reflects:\n")
            f.write("1. Restored central bank credibility under new management\n")
            f.write("2. Return to orthodox monetary policy framework\n")
            f.write("3. Changed investor expectations about policy effectiveness\n")
        else:
            f.write("The lack of structural break may indicate:\n")
            f.write("1. Markets adapted gradually rather than abruptly\n")
            f.write("2. Other factors dominated the BIST100-rate relationship\n")
            f.write("3. Policy transmission mechanism remained fundamentally similar\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"   ✅ Saved: {output_path}")


if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("H3: STRUCTURAL BREAK ANALYSIS")
    print("="*80)
    print("\nDSA210 Term Project: Digital Panic and Market Dynamics")
    print("Student: Simla Tükenmez (32613)\n")
    
    # Load data
    df = load_data()
    
    # Run H3 test
    results = h3_structural_break_test(df)
    
    print("\n" + "="*80)
    print("✅ H3 ANALYSIS COMPLETE!")
    print("="*80)
    
    print("\nOutput Files:")
    print("   1. h3_structural_break.png - 4-panel visualization")
    print("   2. h3_results.txt - Detailed results")
    
    print("\nQuick Summary:")
    print(f"   Pre-period β₁:    {results['beta1_pre']:>8.4f}")
    print(f"   Post-period β₁:   {results['beta1_post']:>8.4f}")
    print(f"   Chow F-stat:      {results['chow_stat']:>8.4f}")
    print(f"   P-value:          {results['p_value_chow']:>8.4f}")
    print(f"   Result:           {'✅ STRUCTURAL BREAK CONFIRMED' if results['structural_break'] else '❌ NO STRUCTURAL BREAK'}")
    
    if results['structural_break']:
        print("\nCONCLUSION:")
        print("   H3 is SUPPORTED by the data.")
        print("   June 2023 policy change created structural break.")
        print("   Market-rate relationship changed significantly.")
    else:
        print("\nCONCLUSION:")
        print("   H3 is NOT supported.")
        print("   No significant structural break detected.")
        print("   Relationship remained stable across periods.")
