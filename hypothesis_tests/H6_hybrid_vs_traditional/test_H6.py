"""
H6: Hybrid Model Comparison
Compares traditional vs digital-enhanced prediction models

Author: Simla Tükenmez
Student ID: 32613
Course: DSA210 - Fall 2025-2026

Hypothesis: Adding digital panic indicators (Google Trends, Sentiment) 
improves volatility prediction beyond traditional indicators alone

Method: Compare R² of three models:
1. Traditional: Funding Cost + USD/TRY
2. Digital: Google Trends + Sentiment
3. Hybrid: Traditional + Digital
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load master dataset"""
    print("Loading data...")
    df = pd.read_excel('master_data_with_sentiment.xlsx')
    df['date'] = pd.to_datetime(df['date'])
    print(f"   ✅ {len(df)} rows loaded\n")
    return df

def prepare_volatility_data(df):
    """Prepare volatility measure and predictors"""
    
    # Target: Forward-looking volatility (next 7 days)
    df['Volatility_Forward_7d'] = df['BIST100_Return'].shift(-7).rolling(window=7).std()
    
    # Traditional predictors
    df['USD_TRY_Change'] = df['USD_TRY'].pct_change() * 100
    df['Funding_Cost_Change'] = df['Funding_Cost'].diff()
    
    # Digital predictors (already in data)
    # - Digital_Panic_Index (from H4)
    # - Sentiment_Score
    
    # Create Digital Panic if not exists
    if 'Digital_Panic_Index' not in df.columns:
        trends = ['dolar_kuru', 'enflasyon']
        for col in trends:
            if col in df.columns:
                df[f'{col}_normalized'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        normalized_cols = [f'{col}_normalized' for col in trends if f'{col}_normalized' in df.columns]
        if normalized_cols:
            df['Digital_Panic_Index'] = df[normalized_cols].mean(axis=1)
    
    return df

def h6_hybrid_model_comparison(df):
    """Compare traditional vs digital vs hybrid models"""
    
    print("="*80)
    print("H6: HYBRID MODEL COMPARISON")
    print("="*80)
    
    # Prepare data
    df = prepare_volatility_data(df)
    
    # Define predictor sets
    traditional_vars = ['USD_TRY_Change', 'Funding_Cost_Change']
    digital_vars = ['Digital_Panic_Index', 'Sentiment_Score']
    hybrid_vars = traditional_vars + digital_vars
    
    # Create analysis dataset
    all_vars = ['Volatility_Forward_7d'] + hybrid_vars
    analysis_df = df[all_vars].dropna()
    
    print(f"\nAnalysis Sample:")
    print(f"   Observations: {len(analysis_df)}")
    print(f"   Target: Forward 7-day volatility")
    
    print(f"\nVariable Sets:")
    print(f"   Traditional: {traditional_vars}")
    print(f"   Digital:     {digital_vars}")
    print(f"   Hybrid:      {hybrid_vars}")
    
    # Prepare X and y
    y = analysis_df['Volatility_Forward_7d'].values
    
    # Model 1: Traditional only
    X_trad = analysis_df[traditional_vars].values
    model_trad = LinearRegression()
    model_trad.fit(X_trad, y)
    y_pred_trad = model_trad.predict(X_trad)
    r2_trad = r2_score(y, y_pred_trad)
    rmse_trad = np.sqrt(mean_squared_error(y, y_pred_trad))
    
    # Model 2: Digital only
    X_dig = analysis_df[digital_vars].values
    model_dig = LinearRegression()
    model_dig.fit(X_dig, y)
    y_pred_dig = model_dig.predict(X_dig)
    r2_dig = r2_score(y, y_pred_dig)
    rmse_dig = np.sqrt(mean_squared_error(y, y_pred_dig))
    
    # Model 3: Hybrid
    X_hyb = analysis_df[hybrid_vars].values
    model_hyb = LinearRegression()
    model_hyb.fit(X_hyb, y)
    y_pred_hyb = model_hyb.predict(X_hyb)
    r2_hyb = r2_score(y, y_pred_hyb)
    rmse_hyb = np.sqrt(mean_squared_error(y, y_pred_hyb))
    
    # Results
    print(f"\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    print(f"\nMODEL 1: TRADITIONAL (Funding Cost + USD/TRY)")
    print(f"   R²:   {r2_trad:>8.4f}")
    print(f"   RMSE: {rmse_trad:>8.4f}%")
    print(f"   Predictors: {len(traditional_vars)}")
    
    print(f"\nMODEL 2: DIGITAL (Google Trends + Sentiment)")
    print(f"   R²:   {r2_dig:>8.4f}")
    print(f"   RMSE: {rmse_dig:>8.4f}%")
    print(f"   Predictors: {len(digital_vars)}")
    
    print(f"\nMODEL 3: HYBRID (Traditional + Digital)")
    print(f"   R²:   {r2_hyb:>8.4f}")
    print(f"   RMSE: {rmse_hyb:>8.4f}%")
    print(f"   Predictors: {len(hybrid_vars)}")
    
    # Improvement analysis
    print(f"\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)
    
    r2_improvement = r2_hyb - r2_trad
    r2_pct_improvement = (r2_improvement / r2_trad * 100) if r2_trad > 0 else 0
    
    print(f"\nHybrid vs Traditional:")
    print(f"   ΔR² = {r2_improvement:>8.4f} ({r2_pct_improvement:>+6.1f}%)")
    print(f"   ΔRMSE = {rmse_hyb - rmse_trad:>8.4f}%")
    
    if r2_hyb > r2_trad:
        print(f"   ✅ Hybrid model is BETTER")
    else:
        print(f"   ❌ Hybrid model is NOT better")
    
    # F-test for nested models
    print(f"\n" + "="*80)
    print("F-TEST FOR NESTED MODELS")
    print("="*80)
    
    # F-test: Does adding digital variables improve fit?
    n = len(y)
    k_trad = len(traditional_vars)
    k_hyb = len(hybrid_vars)
    
    rss_trad = np.sum((y - y_pred_trad)**2)
    rss_hyb = np.sum((y - y_pred_hyb)**2)
    
    f_stat = ((rss_trad - rss_hyb) / (k_hyb - k_trad)) / (rss_hyb / (n - k_hyb - 1))
    p_value_f = 1 - stats.f.cdf(f_stat, k_hyb - k_trad, n - k_hyb - 1)
    
    print(f"\nTesting: Does adding digital variables improve prediction?")
    print(f"  F-statistic: {f_stat:>8.4f}")
    print(f"  P-value:     {p_value_f:>8.4f}")
    
    alpha = 0.05
    if p_value_f < alpha:
        print(f"\n  ✅ SIGNIFICANT improvement (p < {alpha})")
        print(f"  → Digital variables add meaningful predictive power")
    else:
        print(f"\n  ❌ NOT significant (p > {alpha})")
        print(f"  → Digital variables don't significantly improve prediction")
    
    # Decision
    print(f"\n" + "="*80)
    print("STATISTICAL DECISION")
    print("="*80)
    
    h6_supported = (r2_hyb > r2_trad) and (p_value_f < alpha)
    
    if h6_supported:
        print(f"\n✅ H6 is SUPPORTED")
        print(f"\nConclusion: Hybrid model (Traditional + Digital) SIGNIFICANTLY")
        print(f"outperforms traditional model alone.")
        print(f"\nDigital panic indicators provide INCREMENTAL predictive value")
        print(f"beyond traditional economic indicators.")
    else:
        print(f"\n❌ H6 is NOT SUPPORTED")
        print(f"\nConclusion: Adding digital indicators does NOT significantly")
        print(f"improve volatility prediction.")
        print(f"\nTraditional indicators (funding cost, USD/TRY) capture most")
        print(f"of the predictable variation in volatility.")
    
    # Coefficient analysis
    print(f"\n" + "="*80)
    print("HYBRID MODEL COEFFICIENTS")
    print("="*80)
    
    coef_df = pd.DataFrame({
        'Variable': hybrid_vars,
        'Coefficient': model_hyb.coef_
    })
    coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)
    
    print(f"\nRanked by Absolute Magnitude:")
    for idx, row in coef_df.iterrows():
        print(f"  {row['Variable']:25s}: {row['Coefficient']:>10.6f}")
    
    # Visualization
    print(f"\nCreating visualizations...")
    visualize_h6(y, y_pred_trad, y_pred_dig, y_pred_hyb,
                r2_trad, r2_dig, r2_hyb,
                rmse_trad, rmse_dig, rmse_hyb,
                coef_df, p_value_f)
    
    # Save results
    save_h6_results(analysis_df, traditional_vars, digital_vars,
                   r2_trad, r2_dig, r2_hyb,
                   rmse_trad, rmse_dig, rmse_hyb,
                   coef_df, f_stat, p_value_f, h6_supported)
    
    return {
        'r2_traditional': r2_trad,
        'r2_digital': r2_dig,
        'r2_hybrid': r2_hyb,
        'improvement': r2_improvement,
        'f_stat': f_stat,
        'p_value': p_value_f,
        'supported': h6_supported
    }

def visualize_h6(y_true, y_pred_trad, y_pred_dig, y_pred_hyb,
                r2_trad, r2_dig, r2_hyb,
                rmse_trad, rmse_dig, rmse_hyb,
                coef_df, p_value):
    """Create H6 visualizations"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1-3: Predicted vs Actual for each model
    models = [
        ('Traditional', y_pred_trad, r2_trad, rmse_trad, '#2E86AB', 0, 0),
        ('Digital', y_pred_dig, r2_dig, rmse_dig, '#2A9D8F', 0, 1),
        ('Hybrid', y_pred_hyb, r2_hyb, rmse_hyb, '#E63946', 0, 2)
    ]
    
    for name, y_pred, r2, rmse, color, row, col in models:
        ax = fig.add_subplot(gs[row, col])
        ax.scatter(y_true, y_pred, alpha=0.4, s=20, color=color)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Volatility (%)', fontsize=10)
        ax.set_ylabel('Predicted Volatility (%)', fontsize=10)
        ax.set_title(f'{name} Model\nR²={r2:.4f}, RMSE={rmse:.4f}%', 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Plot 4: R² comparison
    ax4 = fig.add_subplot(gs[1, 0])
    models_names = ['Traditional', 'Digital', 'Hybrid']
    r2_values = [r2_trad, r2_dig, r2_hyb]
    colors_bar = ['#2E86AB', '#2A9D8F', '#E63946']
    
    bars = ax4.bar(models_names, r2_values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for bar, r2 in zip(bars, r2_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{r2:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax4.set_ylabel('R² (Explained Variance)', fontsize=11)
    ax4.set_title('H6: Model Performance Comparison', fontsize=13, fontweight='bold')
    ax4.set_ylim(0, max(r2_values) * 1.2)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: RMSE comparison
    ax5 = fig.add_subplot(gs[1, 1])
    rmse_values = [rmse_trad, rmse_dig, rmse_hyb]
    
    bars = ax5.bar(models_names, rmse_values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for bar, rmse in zip(bars, rmse_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax5.set_ylabel('RMSE (Root Mean Squared Error, %)', fontsize=11)
    ax5.set_title('H6: Prediction Error Comparison', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Coefficient importance (Hybrid model)
    ax6 = fig.add_subplot(gs[1, 2])
    
    colors_coef = ['#2E86AB' if v in ['USD_TRY_Change', 'Funding_Cost_Change'] 
                   else '#2A9D8F' for v in coef_df['Variable']]
    
    bars = ax6.barh(coef_df['Variable'], coef_df['Coefficient'], 
                    color=colors_coef, alpha=0.7, edgecolor='black', linewidth=1)
    ax6.axvline(0, color='black', linewidth=1.5)
    
    ax6.set_xlabel('Coefficient Value', fontsize=11)
    ax6.set_title('H6: Hybrid Model Coefficients\n(Blue=Traditional, Green=Digital)', 
                 fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    
    # Add F-test result
    ax6.text(0.98, 0.02, f'F-test p-value: {p_value:.4f}\n' + 
             ('Improvement: Significant ✓' if p_value < 0.05 else 'Improvement: Not Significant'),
             transform=ax6.transAxes, fontsize=9, 
             ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='yellow' if p_value < 0.05 else 'lightgray', alpha=0.8))
    
    plt.savefig('h6_hybrid_model_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: h6_hybrid_model_comparison.png")
    plt.close()

def save_h6_results(df, trad_vars, dig_vars, r2_t, r2_d, r2_h,
                   rmse_t, rmse_d, rmse_h, coef_df, f_stat, p_val, supported):
    """Save H6 results"""
    
    with open('h6_results.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("H6: HYBRID MODEL COMPARISON - RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Student: Simla Tükenmez (32613)\n")
        f.write(f"Course: DSA210 - Fall 2025-2026\n\n")
        
        f.write("="*80 + "\n")
        f.write("RESEARCH QUESTION\n")
        f.write("="*80 + "\n\n")
        f.write("Do digital panic indicators (Google Trends, Sentiment) provide\n")
        f.write("incremental predictive value beyond traditional economic indicators?\n\n")
        
        f.write("="*80 + "\n")
        f.write("METHODOLOGY\n")
        f.write("="*80 + "\n\n")
        f.write("Target Variable: Forward 7-day volatility\n\n")
        f.write("Model 1 - Traditional:\n")
        f.write(f"  Predictors: {', '.join(trad_vars)}\n\n")
        f.write("Model 2 - Digital:\n")
        f.write(f"  Predictors: {', '.join(dig_vars)}\n\n")
        f.write("Model 3 - Hybrid:\n")
        f.write(f"  Predictors: All of the above\n\n")
        f.write(f"Sample Size: {len(df)} observations\n\n")
        
        f.write("="*80 + "\n")
        f.write("MODEL PERFORMANCE\n")
        f.write("="*80 + "\n\n")
        f.write(f"Traditional Model:\n")
        f.write(f"  R²:   {r2_t:>8.4f}\n")
        f.write(f"  RMSE: {rmse_t:>8.4f}%\n\n")
        
        f.write(f"Digital Model:\n")
        f.write(f"  R²:   {r2_d:>8.4f}\n")
        f.write(f"  RMSE: {rmse_d:>8.4f}%\n\n")
        
        f.write(f"Hybrid Model:\n")
        f.write(f"  R²:   {r2_h:>8.4f}\n")
        f.write(f"  RMSE: {rmse_h:>8.4f}%\n\n")
        
        f.write("="*80 + "\n")
        f.write("IMPROVEMENT ANALYSIS\n")
        f.write("="*80 + "\n\n")
        improvement = r2_h - r2_t
        pct_improvement = (improvement / r2_t * 100) if r2_t > 0 else 0
        f.write(f"Hybrid vs Traditional:\n")
        f.write(f"  ΔR² = {improvement:>8.4f} ({pct_improvement:>+6.1f}%)\n")
        f.write(f"  ΔRMSE = {rmse_h - rmse_t:>8.4f}%\n\n")
        
        f.write(f"F-Test (Nested Models):\n")
        f.write(f"  F-statistic: {f_stat:>8.4f}\n")
        f.write(f"  P-value:     {p_val:>8.4f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("HYBRID MODEL COEFFICIENTS\n")
        f.write("="*80 + "\n\n")
        for idx, row in coef_df.iterrows():
            f.write(f"  {row['Variable']:25s}: {row['Coefficient']:>10.6f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("CONCLUSION\n")
        f.write("="*80 + "\n\n")
        
        if supported:
            f.write("Decision: H6 is SUPPORTED\n\n")
            f.write(f"The hybrid model SIGNIFICANTLY outperforms the traditional model.\n")
            f.write(f"Adding digital panic indicators improves R² by {improvement:.4f}.\n\n")
            f.write("INTERPRETATION:\n")
            f.write("Digital indicators (Google Trends, social sentiment) capture\n")
            f.write("behavioral/psychological dimensions of market volatility that\n")
            f.write("traditional economic indicators miss. This validates the use\n")
            f.write("of digital data sources in financial prediction models.\n")
        else:
            f.write("Decision: H6 is NOT SUPPORTED\n\n")
            f.write(f"The hybrid model does not significantly outperform traditional.\n")
            f.write(f"P-value ({p_val:.4f}) > 0.05, improvement not statistically significant.\n\n")
            f.write("INTERPRETATION:\n")
            f.write("Traditional economic indicators (funding cost, USD/TRY) already\n")
            f.write("capture most predictable variation in volatility. Digital indicators\n")
            f.write("provide limited incremental value, possibly because they reflect\n")
            f.write("rather than predict market moves.\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print("   ✅ Saved: h6_results.txt")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("H6: HYBRID MODEL COMPARISON")
    print("="*80)
    print("\nDSA210 Term Project: Digital Panic and Market Dynamics")
    print("Student: Simla Tükenmez (32613)\n")
    
    df = load_data()
    results = h6_hybrid_model_comparison(df)
    
    print("\n" + "="*80)
    print("✅ H6 ANALYSIS COMPLETE!")
    print("="*80)
    
    print("\nOutput Files:")
    print("   1. h6_hybrid_model_comparison.png")
    print("   2. h6_results.txt")
    
    print("\nQuick Summary:")
    print(f"   Traditional R²: {results['r2_traditional']:>8.4f}")
    print(f"   Digital R²:     {results['r2_digital']:>8.4f}")
    print(f"   Hybrid R²:      {results['r2_hybrid']:>8.4f}")
    print(f"   Improvement:    {results['improvement']:>8.4f}")
    print(f"   F-test p-value: {results['p_value']:>8.4f}")
    print(f"   Result:         {'✅ SUPPORTED' if results['supported'] else '❌ NOT SUPPORTED'}")
    
    if results['supported']:
        print("\nCONCLUSION: H6 SUPPORTED")
        print("   Digital indicators add significant value")
    else:
        print("\nCONCLUSION: H6 NOT SUPPORTED")
        print("   Traditional indicators sufficient")
    
    print("\n")
