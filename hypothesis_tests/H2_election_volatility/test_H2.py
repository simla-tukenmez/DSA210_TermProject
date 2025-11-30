"""
H2: Election Volatility Test
Tests whether May 2023 elections caused higher market volatility

Author: Simla Tükenmez
Student ID: 32613
Course: DSA210 - Fall 2025-2026

Hypothesis: Election month (May 2023) had significantly higher 
BIST100 volatility compared to non-election periods

Method: Levene's test (variance equality test)
"""

import pandas as pd
import numpy as np
from scipy import stats
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


def h2_election_volatility_test(df):
    """
    Test H2: Election month volatility
    
    Compare BIST100 volatility in:
    - Election month (May 2023)
    - Non-election months (all other months in 2023)
    
    Method: Levene's test for variance equality
    """
    
    print("="*80)
    print("H2: ELECTION VOLATILITY TEST")
    print("="*80)
    
    # Define periods
    election_month = df[(df['date'] >= '2023-05-01') & (df['date'] <= '2023-05-31')]
    
    # Non-election: Rest of 2023 (excluding May)
    non_election = df[
        (df['date'] >= '2023-01-01') & 
        (df['date'] <= '2023-12-31') &
        ~((df['date'] >= '2023-05-01') & (df['date'] <= '2023-05-31'))
    ]
    
    print(f"\nAnalysis Periods:")
    print(f"   Election month (May 2023):     {election_month['date'].min().date()} to {election_month['date'].max().date()}")
    print(f"   Non-election (2023, ex-May):   {non_election['date'].min().date()} to {non_election['date'].max().date()}")
    print(f"\n   Election days:     {len(election_month)}")
    print(f"   Non-election days: {len(non_election)}")
    
    # Get returns (for calculating volatility measures)
    election_returns = election_month['BIST100_Return'].dropna()
    non_election_returns = non_election['BIST100_Return'].dropna()
    
    print(f"\n   Usable returns:")
    print(f"   Election:     {len(election_returns)} days")
    print(f"   Non-election: {len(non_election_returns)} days")
    
    # Descriptive statistics
    print(f"\n" + "="*80)
    print("DESCRIPTIVE STATISTICS - RETURNS")
    print("="*80)
    
    print(f"\nElection Month (May 2023):")
    print(f"   Mean return:   {election_returns.mean():>8.4f}%")
    print(f"   Std Dev:       {election_returns.std():>8.4f}%")
    print(f"   Min:           {election_returns.min():>8.4f}%")
    print(f"   Max:           {election_returns.max():>8.4f}%")
    print(f"   Range:         {election_returns.max() - election_returns.min():>8.4f}%")
    
    print(f"\nNon-Election Months (2023, excluding May):")
    print(f"   Mean return:   {non_election_returns.mean():>8.4f}%")
    print(f"   Std Dev:       {non_election_returns.std():>8.4f}%")
    print(f"   Min:           {non_election_returns.min():>8.4f}%")
    print(f"   Max:           {non_election_returns.max():>8.4f}%")
    print(f"   Range:         {non_election_returns.max() - non_election_returns.min():>8.4f}%")
    
    # Volatility comparison
    print(f"\nVolatility Comparison:")
    vol_ratio = election_returns.std() / non_election_returns.std()
    print(f"   Election volatility / Non-election volatility = {vol_ratio:.4f}")
    
    if vol_ratio > 1.0:
        print(f"   → Election month was {(vol_ratio-1)*100:.1f}% MORE volatile")
    else:
        print(f"   → Election month was {(1-vol_ratio)*100:.1f}% LESS volatile")
    
    # Context - Election dates
    print(f"\nElection Context:")
    print(f"   Round 1: May 14, 2023 (Sunday)")
    print(f"   Round 2: May 28, 2023 (Sunday)")
    print(f"   → Markets open Mon-Fri, so impact on May 15 and May 29")
    
    # Check specific election days
    may15 = df[df['date'] == '2023-05-15']
    may29 = df[df['date'] == '2023-05-29']
    
    if len(may15) > 0:
        print(f"\n   May 15 (after Round 1):")
        print(f"     BIST100 Return: {may15['BIST100_Return'].values[0]:.2f}%")
    
    if len(may29) > 0:
        print(f"   May 29 (after Round 2):")
        print(f"     BIST100 Return: {may29['BIST100_Return'].values[0]:.2f}%")
    
    # Levene's Test for Variance Equality
    print(f"\n" + "="*80)
    print("LEVENE'S TEST FOR VARIANCE EQUALITY")
    print("="*80)
    
    print(f"\nHypotheses:")
    print(f"   H0 (Null):        Variance(election) = Variance(non-election)")
    print(f"   H2 (Alternative): Variance(election) > Variance(non-election)")
    
    # Levene's test (center='median' is more robust)
    levene_stat, p_value_two = stats.levene(election_returns, non_election_returns, center='median')
    
    # One-tailed p-value (testing if election variance > non-election variance)
    if election_returns.var() > non_election_returns.var():
        p_value_one = p_value_two / 2
    else:
        p_value_one = 1 - (p_value_two / 2)
    
    print(f"\nTest Results:")
    print(f"   Levene statistic:     {levene_stat:>8.4f}")
    print(f"   P-value (two-tailed): {p_value_two:>8.4f}")
    print(f"   P-value (one-tailed): {p_value_one:>8.4f}")
    
    print(f"\nStatistical Decision at α = 0.05:")
    if p_value_one < 0.05:
        print(f"   ✅ REJECT H0")
        print(f"   → Election month had SIGNIFICANTLY HIGHER volatility")
        print(f"   → Evidence SUPPORTS hypothesis H2")
    elif p_value_one < 0.10:
        print(f"   MARGINALLY SIGNIFICANT at α = 0.10")
        print(f"   → Election month volatility higher at 10% level")
        print(f"   → Evidence provides MODERATE SUPPORT for H2")
    else:
        print(f"   ❌ FAIL TO REJECT H0")
        print(f"   → Election month volatility NOT significantly different")
        print(f"   → Insufficient evidence to support H2")
    
    # F-test for variance ratio (alternative test)
    print(f"\n" + "="*80)
    print("F-TEST FOR VARIANCE RATIO (Robustness Check)")
    print("="*80)
    
    var_election = election_returns.var()
    var_non_election = non_election_returns.var()
    
    f_stat = var_election / var_non_election
    df1 = len(election_returns) - 1
    df2 = len(non_election_returns) - 1
    
    # One-tailed F-test p-value
    p_value_f = 1 - stats.f.cdf(f_stat, df1, df2)
    
    print(f"\nF-test Results:")
    print(f"   F-statistic:          {f_stat:>8.4f}")
    print(f"   Degrees of freedom:   df1={df1}, df2={df2}")
    print(f"   P-value (one-tailed): {p_value_f:>8.4f}")
    
    if p_value_f < 0.05:
        print(f"   ✅ F-test also shows SIGNIFICANT difference in variance")
    else:
        print(f"   ❌ F-test does NOT show significant difference")
    
    # Additional analysis - rolling volatility
    print(f"\n" + "="*80)
    print("ADDITIONAL CONTEXT")
    print("="*80)
    
    # Absolute returns (proxy for volatility)
    election_abs_returns = election_returns.abs()
    non_election_abs_returns = non_election_returns.abs()
    
    print(f"\nAbsolute Returns (volatility proxy):")
    print(f"   Election mean |return|:     {election_abs_returns.mean():.4f}%")
    print(f"   Non-election mean |return|: {non_election_abs_returns.mean():.4f}%")
    print(f"   Ratio: {election_abs_returns.mean() / non_election_abs_returns.mean():.4f}")
    
    # Daily returns range
    print(f"\nIntraday Range:")
    print(f"   Election range:     {election_returns.max() - election_returns.min():.2f}%")
    print(f"   Non-election range: {non_election_returns.max() - non_election_returns.min():.2f}%")
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    visualize_election_volatility(df, election_month, non_election, 
                                  election_returns, non_election_returns,
                                  p_value_one, levene_stat)
    
    # Save results
    save_h2_results(election_returns, non_election_returns, 
                   levene_stat, p_value_one, f_stat, p_value_f)
    
    return {
        'election_std': election_returns.std(),
        'non_election_std': non_election_returns.std(),
        'variance_ratio': f_stat,
        'levene_stat': levene_stat,
        'p_value': p_value_one,
        'significant': p_value_one < 0.05
    }


def visualize_election_volatility(df, election_df, non_election_df, 
                                  election_returns, non_election_returns,
                                  p_value, levene_stat):
    """Create comprehensive visualizations for election volatility"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    # Plot 1: BIST100 returns over 2023 with election month highlighted
    ax1 = axes[0, 0]
    year_2023 = df[(df['date'] >= '2023-01-01') & (df['date'] <= '2023-12-31')]
    
    # Color code by period
    colors = ['#E63946' if ((d >= pd.Timestamp('2023-05-01')) and (d <= pd.Timestamp('2023-05-31'))) 
              else '#2E86AB' for d in year_2023['date']]
    
    ax1.bar(year_2023['date'], year_2023['BIST100_Return'], 
           color=colors, alpha=0.7, width=0.8, edgecolor='white', linewidth=0.3)
    ax1.axhline(0, color='black', linewidth=1.5)
    ax1.axvspan(pd.Timestamp('2023-05-01'), pd.Timestamp('2023-05-31'), 
                alpha=0.2, color='red', label='Election Month')
    
    # Mark election days
    ax1.axvline(pd.Timestamp('2023-05-15'), color='red', linestyle='--', 
               linewidth=2, alpha=0.8, label='Round 1 (May 14)')
    ax1.axvline(pd.Timestamp('2023-05-29'), color='darkred', linestyle='--', 
               linewidth=2, alpha=0.8, label='Round 2 (May 28)')
    
    ax1.set_title('BIST100 Daily Returns in 2023 - Election Month Highlighted', 
                 fontsize=13, fontweight='bold', pad=15)
    ax1.set_ylabel('Daily Return (%)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
    
    # Plot 2: Boxplot comparison
    ax2 = axes[0, 1]
    
    returns_data = pd.DataFrame({
        'Returns': list(non_election_returns) + list(election_returns),
        'Period': ['Non-Election']*len(non_election_returns) + ['Election Month']*len(election_returns)
    })
    
    bp = ax2.boxplot([non_election_returns, election_returns], 
                     tick_labels=['Non-Election\n(2023, ex-May)', 'Election Month\n(May 2023)'],
                     patch_artist=True, widths=0.5, showmeans=True)
    
    colors_box = ['#2E86AB', '#E63946']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.axhline(0, color='black', linestyle=':', linewidth=1.5)
    ax2.set_title('Returns Distribution Comparison', fontsize=13, fontweight='bold', pad=15)
    ax2.set_ylabel('Daily Return (%)', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    stats_text = f'Non-Election: σ={non_election_returns.std():.2f}%\n'
    stats_text += f'Election: σ={election_returns.std():.2f}%\n'
    stats_text += f'Levene p={p_value:.4f}'
    
    ax2.text(0.05, 0.95, stats_text,
            transform=ax2.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 3: Histogram comparison
    ax3 = axes[1, 0]
    
    ax3.hist(non_election_returns, bins=30, alpha=0.6, color='#2E86AB', 
            label=f'Non-Election (σ={non_election_returns.std():.2f}%)', edgecolor='black', linewidth=0.5)
    ax3.hist(election_returns, bins=15, alpha=0.6, color='#E63946', 
            label=f'Election (σ={election_returns.std():.2f}%)', edgecolor='black', linewidth=0.5)
    
    ax3.axvline(non_election_returns.mean(), color='#2E86AB', linestyle='--', linewidth=2)
    ax3.axvline(election_returns.mean(), color='#E63946', linestyle='--', linewidth=2)
    ax3.axvline(0, color='black', linestyle=':', linewidth=1)
    
    ax3.set_title('Returns Distribution - Election vs Non-Election', 
                 fontsize=13, fontweight='bold', pad=15)
    ax3.set_xlabel('Daily Return (%)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Rolling volatility (30-day)
    ax4 = axes[1, 1]
    
    year_2023_full = df[(df['date'] >= '2023-01-01') & (df['date'] <= '2023-12-31')].copy()
    year_2023_full['Rolling_Vol'] = year_2023_full['BIST100_Return'].rolling(window=30).std()
    
    ax4.plot(year_2023_full['date'], year_2023_full['Rolling_Vol'], 
            linewidth=2, color='#2E86AB', label='30-day Rolling Volatility')
    ax4.axvspan(pd.Timestamp('2023-05-01'), pd.Timestamp('2023-05-31'), 
                alpha=0.2, color='red', label='Election Month')
    ax4.axhline(year_2023_full['Rolling_Vol'].mean(), color='orange', 
               linestyle='--', linewidth=1.5, label='2023 Average')
    
    ax4.set_title('30-Day Rolling Volatility in 2023', fontsize=13, fontweight='bold', pad=15)
    ax4.set_ylabel('Volatility (Std Dev %)', fontsize=11)
    ax4.set_xlabel('Date', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    plt.tight_layout()
    plt.savefig('h2_election_volatility.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: h2_election_volatility.png")
    plt.close()


def save_h2_results(election_returns, non_election_returns, 
                   levene_stat, p_value_levene, f_stat, p_value_f):
    """Save H2 test results"""
    
    output_path = 'h2_results.txt'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("H2: ELECTION VOLATILITY TEST - RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Student: Simla Tükenmez (32613)\n")
        f.write(f"Course: DSA210 - Fall 2025-2026\n\n")
        
        f.write("="*80 + "\n")
        f.write("RESEARCH QUESTION\n")
        f.write("="*80 + "\n\n")
        f.write("Did the May 2023 Turkish elections cause significantly higher\n")
        f.write("volatility in BIST100 stock index returns?\n\n")
        
        f.write("="*80 + "\n")
        f.write("HYPOTHESES\n")
        f.write("="*80 + "\n\n")
        f.write("H0 (Null):        Variance(election) = Variance(non-election)\n")
        f.write("H2 (Alternative): Variance(election) > Variance(non-election)\n\n")
        
        f.write("="*80 + "\n")
        f.write("METHODOLOGY\n")
        f.write("="*80 + "\n\n")
        f.write("Comparison Periods:\n")
        f.write("  Election period:     May 1-31, 2023\n")
        f.write("  Non-election period: Jan-Apr, Jun-Dec 2023\n\n")
        
        f.write("Election Context:\n")
        f.write("  Round 1: May 14, 2023 (presidential election)\n")
        f.write("  Round 2: May 28, 2023 (runoff election)\n\n")
        
        f.write("Statistical Tests:\n")
        f.write("  Primary: Levene's test (robust to non-normality)\n")
        f.write("  Secondary: F-test for variance ratio\n\n")
        
        f.write("="*80 + "\n")
        f.write("DESCRIPTIVE STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Non-Election Period (2023, excluding May):\n")
        f.write(f"  Observations:  {len(non_election_returns)}\n")
        f.write(f"  Mean return:   {non_election_returns.mean():>8.4f}%\n")
        f.write(f"  Std deviation: {non_election_returns.std():>8.4f}%\n")
        f.write(f"  Variance:      {non_election_returns.var():>8.4f}\n")
        f.write(f"  Min:           {non_election_returns.min():>8.4f}%\n")
        f.write(f"  Max:           {non_election_returns.max():>8.4f}%\n\n")
        
        f.write(f"Election Month (May 2023):\n")
        f.write(f"  Observations:  {len(election_returns)}\n")
        f.write(f"  Mean return:   {election_returns.mean():>8.4f}%\n")
        f.write(f"  Std deviation: {election_returns.std():>8.4f}%\n")
        f.write(f"  Variance:      {election_returns.var():>8.4f}\n")
        f.write(f"  Min:           {election_returns.min():>8.4f}%\n")
        f.write(f"  Max:           {election_returns.max():>8.4f}%\n\n")
        
        f.write(f"Volatility Comparison:\n")
        f.write(f"  Variance ratio (Election/Non-Election): {f_stat:.4f}\n")
        if f_stat > 1:
            f.write(f"  → Election month was {(f_stat-1)*100:.1f}% MORE volatile\n\n")
        else:
            f.write(f"  → Election month was {(1-f_stat)*100:.1f}% LESS volatile\n\n")
        
        f.write("="*80 + "\n")
        f.write("STATISTICAL TEST RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Levene's Test (Primary):\n")
        f.write(f"  Test statistic:       {levene_stat:>8.4f}\n")
        f.write(f"  P-value (one-tailed): {p_value_levene:>8.4f}\n")
        f.write(f"  Significance level:   α = 0.05\n\n")
        
        f.write(f"F-Test (Robustness Check):\n")
        f.write(f"  F-statistic:          {f_stat:>8.4f}\n")
        f.write(f"  P-value (one-tailed): {p_value_f:>8.4f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("CONCLUSION\n")
        f.write("="*80 + "\n\n")
        
        if p_value_levene < 0.05:
            f.write("Decision: REJECT H0 at α = 0.05\n\n")
            f.write("H2 is SUPPORTED. The election month showed SIGNIFICANTLY HIGHER\n")
            f.write("volatility in BIST100 returns compared to non-election periods.\n\n")
            f.write(f"The election month volatility (σ = {election_returns.std():.4f}%) was\n")
            f.write(f"significantly higher than non-election volatility (σ = {non_election_returns.std():.4f}%),\n")
            f.write(f"with p = {p_value_levene:.4f} < 0.05.\n\n")
        else:
            f.write("Decision: FAIL TO REJECT H0 at α = 0.05\n\n")
            f.write("H2 is NOT SUPPORTED. The election month did NOT show significantly\n")
            f.write("higher volatility compared to non-election periods.\n\n")
            f.write(f"Although election volatility (σ = {election_returns.std():.4f}%) differed from\n")
            f.write(f"non-election volatility (σ = {non_election_returns.std():.4f}%), this difference\n")
            f.write(f"was not statistically significant (p = {p_value_levene:.4f}).\n\n")
        
        f.write("="*80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*80 + "\n\n")
        f.write("Political events like elections can increase market uncertainty,\n")
        f.write("leading to higher volatility. This test examined whether the\n")
        f.write("2023 Turkish presidential election created such volatility.\n\n")
        
        if p_value_levene < 0.05:
            f.write("The significant increase in volatility during May 2023 suggests\n")
            f.write("that political uncertainty surrounding the election affected\n")
            f.write("investor behavior, creating more volatile market conditions.\n")
        else:
            f.write("The lack of significant volatility increase may indicate:\n")
            f.write("1. Markets had already priced in election outcomes\n")
            f.write("2. Election results were not surprising to investors\n")
            f.write("3. Other factors dominated market volatility in 2023\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"   ✅ Saved: {output_path}")


if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("H2: ELECTION VOLATILITY ANALYSIS")
    print("="*80)
    print("\nDSA210 Term Project: Digital Panic and Market Dynamics")
    print("Student: Simla Tükenmez (32613)\n")
    
    # Load data
    df = load_data()
    
    # Run H2 test
    results = h2_election_volatility_test(df)
    
    print("\n" + "="*80)
    print("✅ H2 ANALYSIS COMPLETE!")
    print("="*80)
    
    print("\nOutput Files:")
    print("   1. h2_election_volatility.png - 4-panel visualization")
    print("   2. h2_results.txt - Detailed results")
    
    print("\nQuick Summary:")
    print(f"   Non-election volatility: {results['non_election_std']:>8.4f}%")
    print(f"   Election volatility:     {results['election_std']:>8.4f}%")
    print(f"   Variance ratio:          {results['variance_ratio']:>8.4f}")
    print(f"   Levene statistic:        {results['levene_stat']:>8.4f}")
    print(f"   P-value:                 {results['p_value']:>8.4f}")
    print(f"   Result:                  {'✅ SIGNIFICANT' if results['significant'] else '❌ NOT SIGNIFICANT'}")
    
    if results['significant']:
        print("\nCONCLUSION:")
        print("   H2 is SUPPORTED by the data.")
        print("   Election month had significantly higher volatility.")
    else:
        print("\nCONCLUSION:")
        print("   H2 is NOT supported at α = 0.05 level.")
        print("   Election month volatility was not significantly different.")
