"""Evaluation Module - Comprehensive metrics computation and reporting."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.config import *
from src.utils import print_section, print_step, print_metric, save_dataframe


def evaluate_demand_prediction(source='urbanev'):
    """Evaluate Demand Prediction Agent."""
    print_step('Evaluating Demand Prediction Agent...')
    
    metrics_path = os.path.join(MODEL_OUTPUTS_DIR, f'demand_metrics_{source}.csv')
    if not os.path.exists(metrics_path):
        print(f'  ⚠ Metrics not found: {metrics_path}')
        return None
    
    metrics_df = pd.read_csv(metrics_path)
    
    print('\n  Demand Prediction Metrics:')
    print(f'  {"Model":20s} | {"RMSE":>8s} | {"MAE":>8s} | {"R²":>8s}')
    print(f'  {"-"*20} | {"-"*8} | {"-"*8} | {"-"*8}')
    for _, row in metrics_df.iterrows():
        print(f'  {row["model"]:20s} | {row["rmse"]:8.4f} | {row["mae"]:8.4f} | {row["r2"]:8.4f}')
    
    # Plot model comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    models = metrics_df['model'].values
    
    axes[0].bar(models, metrics_df['rmse'], color=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['success']][:len(models)])
    axes[0].set_title('RMSE by Model', fontweight='bold')
    axes[0].set_ylabel('RMSE (lower is better)')
    for i, v in enumerate(metrics_df['rmse']):
        axes[0].text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=9)
    
    axes[1].bar(models, metrics_df['mae'], color=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['success']][:len(models)])
    axes[1].set_title('MAE by Model', fontweight='bold')
    axes[1].set_ylabel('MAE (lower is better)')
    for i, v in enumerate(metrics_df['mae']):
        axes[1].text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=9)
    
    axes[2].bar(models, metrics_df['r2'], color=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['success']][:len(models)])
    axes[2].set_title('R² Score by Model', fontweight='bold')
    axes[2].set_ylabel('R² (higher is better)')
    for i, v in enumerate(metrics_df['r2']):
        axes[2].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=9)
    
    plt.suptitle('Demand Prediction Agent — Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EVALUATION_DIR, f'demand_model_comparison_{source}.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    # Feature importance plot
    fi_path = os.path.join(MODEL_OUTPUTS_DIR, f'feature_importance_{source}.csv')
    if os.path.exists(fi_path):
        fi_df = pd.read_csv(fi_path).head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(fi_df['feature'][::-1], fi_df['importance'][::-1], color=COLOR_PALETTE['primary'])
        ax.set_title('Top 15 Feature Importance (XGBoost)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(os.path.join(EVALUATION_DIR, f'feature_importance_{source}.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
    
    # Predicted vs Actual scatter
    pred_path = os.path.join(MODEL_OUTPUTS_DIR, f'demand_predictions_{source}.csv')
    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)
        if 'predicted_utilization' in pred_df.columns and 'utilization_rate' in pred_df.columns:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(pred_df['utilization_rate'], pred_df['predicted_utilization'],
                      alpha=0.3, s=5, color=COLOR_PALETTE['primary'])
            ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
            ax.set_xlabel('Actual Utilization', fontsize=12)
            ax.set_ylabel('Predicted Utilization', fontsize=12)
            ax.set_title('Predicted vs Actual Utilization (XGBoost)', fontsize=14, fontweight='bold')
            ax.legend()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(EVALUATION_DIR, f'pred_vs_actual_{source}.png'), dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
    
    return metrics_df


def evaluate_tariff_pricing(source='urbanev'):
    """Evaluate Tariff Pricing Agent."""
    print_step('Evaluating Tariff Pricing Agent...')
    
    tariff_metrics_path = os.path.join(MODEL_OUTPUTS_DIR, f'tariff_metrics_{source}.csv')
    if not os.path.exists(tariff_metrics_path):
        print(f'  ⚠ Tariff metrics not found: {tariff_metrics_path}')
        return None
    
    metrics = pd.read_csv(tariff_metrics_path).iloc[0].to_dict()
    
    print('\n  Tariff Pricing Metrics:')
    print(f'    Revenue Gain: {metrics.get("revenue_gain_pct", 0):.2f}%')
    print(f'    Off-Peak Uplift: {metrics.get("off_peak_uplift_pct", 0):.2f}%')
    print(f'    Avg Wait Reduction: {metrics.get("avg_wait_reduction_min", 0):.2f} min')
    
    # Tariff distribution plot
    priced_path = os.path.join(MODEL_OUTPUTS_DIR, f'tariff_pricing_{source}.csv')
    if os.path.exists(priced_path):
        priced_df = pd.read_csv(priced_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Tariff distribution
        axes[0, 0].hist(priced_df['dynamic_tariff'], bins=50, color=COLOR_PALETTE['primary'], alpha=0.7, edgecolor='white')
        axes[0, 0].axvline(x=BASELINE_TARIFF, color=COLOR_PALETTE['danger'], linestyle='--', linewidth=2, label=f'Fixed Baseline ₹{BASELINE_TARIFF}')
        axes[0, 0].set_title('Dynamic Tariff Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Tariff (₹/kWh)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # 2. Revenue comparison by time slot
        if 'revenue_fixed' in priced_df.columns and 'revenue_dynamic' in priced_df.columns:
            rev_by_slot = priced_df.groupby('time_slot').agg(
                fixed=('revenue_fixed', 'sum'),
                dynamic=('revenue_dynamic', 'sum')
            ).reindex(['off_peak', 'shoulder', 'peak'])
            
            x = np.arange(len(rev_by_slot))
            w = 0.35
            axes[0, 1].bar(x - w/2, rev_by_slot['fixed']/1e6, w, label='Fixed', color=COLOR_PALETTE['danger'], alpha=0.7)
            axes[0, 1].bar(x + w/2, rev_by_slot['dynamic']/1e6, w, label='Dynamic', color=COLOR_PALETTE['success'], alpha=0.7)
            axes[0, 1].set_title('Revenue: Fixed vs Dynamic (₹M)', fontweight='bold')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(['Off-Peak', 'Shoulder', 'Peak'])
            axes[0, 1].set_ylabel('Revenue (₹ Millions)')
            axes[0, 1].legend()
        
        # 3. Utilization vs Tariff
        if 'predicted_utilization' in priced_df.columns:
            axes[1, 0].scatter(priced_df['predicted_utilization'], priced_df['dynamic_tariff'],
                             alpha=0.2, s=5, color=COLOR_PALETTE['accent'])
            axes[1, 0].axhline(y=BASELINE_TARIFF, color=COLOR_PALETTE['danger'], linestyle='--', label='Fixed Rate')
            axes[1, 0].axvline(x=0.8, color=COLOR_PALETTE['danger'], linestyle=':', alpha=0.5, label='Surge Threshold')
            axes[1, 0].axvline(x=0.3, color=COLOR_PALETTE['success'], linestyle=':', alpha=0.5, label='Discount Threshold')
            axes[1, 0].set_title('Utilization vs Dynamic Tariff', fontweight='bold')
            axes[1, 0].set_xlabel('Predicted Utilization')
            axes[1, 0].set_ylabel('Dynamic Tariff (₹/kWh)')
            axes[1, 0].legend(fontsize=8)
        
        # 4. Tier distribution pie
        if 'tier' in priced_df.columns:
            tier_counts = priced_df['tier'].value_counts()
            tier_colors = [COLOR_PALETTE['peak'], COLOR_PALETTE['danger'], COLOR_PALETTE['primary'],
                          COLOR_PALETTE['accent'], COLOR_PALETTE['success']]
            axes[1, 1].pie(tier_counts.values, labels=tier_counts.index, autopct='%1.1f%%',
                          colors=tier_colors[:len(tier_counts)], startangle=90)
            axes[1, 1].set_title('Pricing Tier Distribution', fontweight='bold')
        
        plt.suptitle('Tariff Pricing Agent — Evaluation Dashboard', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(EVALUATION_DIR, f'tariff_evaluation_{source}.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
    
    return metrics


def evaluate_monitoring(source='urbanev'):
    """Evaluate Monitoring & Learning Agent."""
    print_step('Evaluating Monitoring & Learning Agent...')
    
    episodes_path = os.path.join(MODEL_OUTPUTS_DIR, f'monitoring_episodes_{source}.csv')
    if not os.path.exists(episodes_path):
        print(f'  ⚠ Episode data not found: {episodes_path}')
        return None
    
    episodes_df = pd.read_csv(episodes_path)
    
    print('\n  Monitoring Agent Performance Across Episodes:')
    print(f'  {"Ep":>3s} | {"Revenue":>12s} | {"Gain%":>8s} | {"Util":>6s} | {"Cong%":>6s} | {"₹/kWh":>6s} | {"Wait":>6s}')
    print(f'  {"-"*3} | {"-"*12} | {"-"*8} | {"-"*6} | {"-"*6} | {"-"*6} | {"-"*6}')
    for _, row in episodes_df.iterrows():
        print(f'  {int(row["episode"]):3d} | ₹{row["total_revenue"]:11,.0f} | {row["revenue_gain_pct"]:+7.1f}% | {row["avg_utilization"]:5.3f} | {row["congestion_rate"]:5.1f}% | ₹{row["revenue_per_kwh"]:5.2f} | {row["avg_wait_time"]:5.2f}')
    
    # Plot episode progression
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    eps = episodes_df['episode']
    
    # 1. Revenue trajectory
    axes[0, 0].plot(eps, episodes_df['total_revenue']/1e6, 'o-', color=COLOR_PALETTE['success'], linewidth=2, markersize=8)
    axes[0, 0].fill_between(eps, episodes_df['baseline_revenue'].values/1e6, episodes_df['total_revenue'].values/1e6,
                            alpha=0.2, color=COLOR_PALETTE['success'])
    axes[0, 0].axhline(y=episodes_df['baseline_revenue'].mean()/1e6, color=COLOR_PALETTE['danger'], linestyle='--', label='Fixed Baseline')
    axes[0, 0].set_title('Revenue Trajectory (₹M)', fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Revenue (₹ Millions)')
    axes[0, 0].legend()
    
    # 2. Congestion rate
    axes[0, 1].plot(eps, episodes_df['congestion_rate'], 'o-', color=COLOR_PALETTE['danger'], linewidth=2, markersize=8)
    axes[0, 1].axhline(y=15, color=COLOR_PALETTE['warning'], linestyle='--', alpha=0.5, label='Target (<15%)')
    axes[0, 1].set_title('Congestion Rate Trajectory', fontweight='bold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Congestion Rate (%)')
    axes[0, 1].legend()
    
    # 3. Pricing efficiency
    axes[1, 0].plot(eps, episodes_df['revenue_per_kwh'], 'o-', color=COLOR_PALETTE['primary'], linewidth=2, markersize=8)
    axes[1, 0].axhline(y=BASELINE_TARIFF, color=COLOR_PALETTE['danger'], linestyle='--', label=f'Fixed ₹{BASELINE_TARIFF}/kWh')
    axes[1, 0].set_title('Pricing Efficiency Score (₹/kWh)', fontweight='bold')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Revenue per kWh (₹)')
    axes[1, 0].legend()
    
    # 4. Wait time
    axes[1, 1].plot(eps, episodes_df['avg_wait_time'], 'o-', color=COLOR_PALETTE['secondary'], linewidth=2, markersize=8, label='Dynamic')
    axes[1, 1].plot(eps, episodes_df['baseline_avg_wait'], 's--', color=COLOR_PALETTE['danger'], linewidth=2, markersize=6, label='Baseline (Fixed)')
    axes[1, 1].set_title('Average Wait Time', fontweight='bold')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Wait Time (minutes)')
    axes[1, 1].legend()
    
    plt.suptitle('Monitoring & Learning Agent — Feedback Loop Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EVALUATION_DIR, f'monitoring_evaluation_{source}.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    # Parameter evolution plot
    param_path = os.path.join(MODEL_OUTPUTS_DIR, f'parameter_evolution_{source}.csv')
    if os.path.exists(param_path):
        param_df = pd.read_csv(param_path)
        fig, ax = plt.subplots(figsize=(12, 6))
        for col in ['surge_multiplier', 'critical_multiplier', 'discount_multiplier', 'elasticity']:
            if col in param_df.columns:
                ax.plot(range(len(param_df)), param_df[col], 'o-', linewidth=2, markersize=6, label=col)
        ax.set_title('Pricing Parameter Evolution Across Episodes', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Parameter Value')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(EVALUATION_DIR, f'parameter_evolution_{source}.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
    
    return episodes_df


def generate_summary_report(source='urbanev'):
    """Generate comprehensive summary report CSV."""
    print_step('Generating summary report...')
    
    summary = {}
    
    # Demand prediction
    metrics_path = os.path.join(MODEL_OUTPUTS_DIR, f'demand_metrics_{source}.csv')
    if os.path.exists(metrics_path):
        dm = pd.read_csv(metrics_path)
        best = dm.loc[dm['r2'].idxmax()]
        summary['best_demand_model'] = best['model']
        summary['best_rmse'] = best['rmse']
        summary['best_mae'] = best['mae']
        summary['best_r2'] = best['r2']
    
    # Tariff pricing
    tariff_path = os.path.join(MODEL_OUTPUTS_DIR, f'tariff_metrics_{source}.csv')
    if os.path.exists(tariff_path):
        tm = pd.read_csv(tariff_path).iloc[0]
        summary['revenue_gain_pct'] = tm.get('revenue_gain_pct', 0)
        summary['off_peak_uplift_pct'] = tm.get('off_peak_uplift_pct', 0)
        summary['avg_wait_reduction'] = tm.get('avg_wait_reduction_min', 0)
    
    # Monitoring
    episodes_path = os.path.join(MODEL_OUTPUTS_DIR, f'monitoring_episodes_{source}.csv')
    if os.path.exists(episodes_path):
        ep = pd.read_csv(episodes_path)
        summary['final_revenue_per_kwh'] = ep.iloc[-1]['revenue_per_kwh']
        summary['final_congestion_rate'] = ep.iloc[-1]['congestion_rate']
        summary['total_revenue_all_episodes'] = ep['total_revenue'].sum()
        summary['avg_revenue_gain'] = ep['revenue_gain_pct'].mean()
    
    summary_df = pd.DataFrame([summary])
    save_dataframe(summary_df, os.path.join(EVALUATION_DIR, f'summary_report_{source}.csv'), 'Summary report')
    
    print('\n  ===== FINAL SUMMARY =====')
    for k, v in summary.items():
        if isinstance(v, float):
            print(f'    {k}: {v:.4f}')
        else:
            print(f'    {k}: {v}')
    
    return summary_df


def run_evaluation(source='urbanev'):
    """Run the full evaluation pipeline."""
    print_section('PHASE 6: EVALUATION & REPORTING')
    
    evaluate_demand_prediction(source)
    evaluate_tariff_pricing(source)
    evaluate_monitoring(source)
    summary = generate_summary_report(source)
    
    print(f'\n  ✓ All evaluation outputs saved to {EVALUATION_DIR}')
    return summary


if __name__ == '__main__':
    run_evaluation()
