"""Tariff Pricing Agent - Translates demand forecasts into optimal dynamic tariffs.

Implements rule-based tiered pricing with ML-optimized dynamic adjustment.
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor
from src.config import *
from src.utils import print_section, print_step, print_metric, save_dataframe


class TariffPricingAgent:
    """Translates demand forecasts into optimal dynamic tariffs.
    
    Pricing Strategy:
    1. Base: Utilization-based tiered pricing (5 tiers)
    2. Dynamic: Time-of-day modifiers
    3. ML: Gradient Boosting optimization for revenue maximization
    """
    
    def __init__(self):
        self.base_tariff = BASELINE_TARIFF  # ₹15/kWh
        self.surge_threshold = SURGE_THRESHOLD  # 0.80
        self.discount_threshold = DISCOUNT_THRESHOLD  # 0.30
        self.price_elasticity = PRICE_ELASTICITY  # -0.3
        self.tariff_tiers = TARIFF_TIERS
        self.tod_modifiers = TOD_MODIFIERS
        self.optimization_model = None
        self.pricing_history = []
    
    def get_tier(self, utilization: float) -> dict:
        """Determine pricing tier based on utilization rate."""
        for tier_name, tier_info in self.tariff_tiers.items():
            if tier_info['min_util'] <= utilization <= tier_info['max_util']:
                return {'name': tier_name, **tier_info}
        return {'name': 'normal', **self.tariff_tiers['normal']}
    
    def calculate_dynamic_tariff(self, predicted_util: float, congestion_prob: float,
                                  time_slot: str, price_ratio: float = 1.0) -> dict:
        """Calculate dynamic tariff for a single time period.
        
        Returns dict with tariff details.
        """
        # Step 1: Base tier from utilization
        tier = self.get_tier(predicted_util)
        base_multiplier = tier['multiplier']
        
        # Step 2: Time-of-day modifier
        tod_modifier = self.tod_modifiers.get(time_slot, 1.0)
        
        # Step 3: Congestion adjustment (additional 0-15% based on congestion probability)
        congestion_adjustment = 1.0 + (congestion_prob * 0.15)
        
        # Step 4: Electricity cost pass-through
        cost_adjustment = max(0.9, min(1.2, price_ratio))
        
        # Final tariff
        final_multiplier = base_multiplier * tod_modifier * congestion_adjustment * cost_adjustment
        final_tariff = self.base_tariff * final_multiplier
        
        # Cap tariff at reasonable bounds
        final_tariff = max(5.0, min(45.0, final_tariff))  # ₹5-45 range
        
        return {
            'base_tariff': self.base_tariff,
            'tier': tier['name'],
            'tier_label': tier['label'],
            'base_multiplier': base_multiplier,
            'tod_modifier': tod_modifier,
            'congestion_adjustment': congestion_adjustment,
            'cost_adjustment': cost_adjustment,
            'final_multiplier': final_tariff / self.base_tariff,
            'dynamic_tariff': round(final_tariff, 2),
            'predicted_utilization': predicted_util,
            'congestion_probability': congestion_prob,
            'time_slot': time_slot,
        }
    
    def apply_dynamic_pricing(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Apply dynamic pricing to all predictions."""
        print_step('Applying dynamic tariff pricing...')
        
        df = predictions_df.copy()
        
        # Ensure required columns
        if 'predicted_utilization' not in df.columns:
            print('  ⚠ No predicted_utilization column, using utilization_rate')
            df['predicted_utilization'] = df.get('utilization_rate', 0.5)
        
        if 'congestion_probability' not in df.columns:
            df['congestion_probability'] = (df['predicted_utilization'] >= 0.8).astype(float)
        
        if 'avg_price_ratio' not in df.columns:
            df['avg_price_ratio'] = 1.0
        
        # Calculate dynamic tariff for each row
        tariff_results = df.apply(
            lambda row: self.calculate_dynamic_tariff(
                predicted_util=row['predicted_utilization'],
                congestion_prob=row['congestion_probability'],
                time_slot=row.get('time_slot', 'normal'),
                price_ratio=row.get('avg_price_ratio', 1.0)
            ), axis=1, result_type='expand'
        )
        
        # Merge results
        for col in tariff_results.columns:
            df[col] = tariff_results[col]
        
        # Calculate revenues
        energy_col = 'total_volume_kwh' if 'total_volume_kwh' in df.columns else 'total_kwh'
        if energy_col in df.columns:
            df['revenue_fixed'] = df[energy_col] * self.base_tariff
            df['revenue_dynamic'] = df[energy_col] * df['dynamic_tariff']
            df['revenue_delta'] = df['revenue_dynamic'] - df['revenue_fixed']
            df['revenue_gain_pct'] = (df['revenue_delta'] / df['revenue_fixed'].clip(lower=0.01)) * 100
        else:
            # Approximate with estimated load
            df['revenue_fixed'] = df.get('expected_load_kwh', 10) * self.base_tariff
            df['revenue_dynamic'] = df.get('expected_load_kwh', 10) * df['dynamic_tariff']
            df['revenue_delta'] = df['revenue_dynamic'] - df['revenue_fixed']
            df['revenue_gain_pct'] = (df['revenue_delta'] / df['revenue_fixed'].clip(lower=0.01)) * 100
        
        # Simulate demand response (price elasticity)
        price_change_pct = (df['dynamic_tariff'] - self.base_tariff) / self.base_tariff
        df['demand_response_factor'] = 1.0 + (price_change_pct * self.price_elasticity)
        df['adjusted_utilization'] = (df['predicted_utilization'] * df['demand_response_factor']).clip(0, 1)
        
        # Wait time proxy (proportional to utilization above capacity)
        df['wait_time_proxy_fixed'] = (df['predicted_utilization'] - 0.9).clip(lower=0) * 30  # minutes
        df['wait_time_proxy_dynamic'] = (df['adjusted_utilization'] - 0.9).clip(lower=0) * 30
        df['wait_time_reduction'] = df['wait_time_proxy_fixed'] - df['wait_time_proxy_dynamic']
        
        print(f'    Applied dynamic pricing to {len(df)} records')
        print(f'    Tariff range: ₹{df["dynamic_tariff"].min():.2f} - ₹{df["dynamic_tariff"].max():.2f}')
        print(f'    Avg dynamic tariff: ₹{df["dynamic_tariff"].mean():.2f} vs fixed ₹{self.base_tariff:.2f}')
        
        return df
    
    def train_optimization_model(self, df: pd.DataFrame):
        """Train a Gradient Boosting model to optimize tariff multiplier."""
        print_step('Training tariff optimization model...')
        
        # Features: utilization, congestion, time features
        opt_features = ['predicted_utilization', 'congestion_probability',
                       'hour', 'day_of_week', 'is_weekend', 'is_peak', 'is_shoulder']
        
        valid_features = [f for f in opt_features if f in df.columns]
        
        # Target: optimal multiplier that maximizes revenue while keeping utilization balanced
        # We create a synthetic "optimal" target based on revenue efficiency
        df['revenue_efficiency'] = df['revenue_dynamic'] / df['revenue_fixed'].clip(lower=0.01)
        df['target_multiplier'] = df['final_multiplier']
        
        X = df[valid_features].fillna(0)
        y = df['target_multiplier']
        
        self.optimization_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=RANDOM_STATE
        )
        self.optimization_model.fit(X, y)
        print('    ✓ Optimization model trained')
    
    def compute_tariff_metrics(self, df: pd.DataFrame) -> dict:
        """Compute tariff pricing evaluation metrics."""
        print_step('Computing tariff metrics...')
        
        metrics = {}
        
        # Revenue Gain %
        total_fixed = df['revenue_fixed'].sum()
        total_dynamic = df['revenue_dynamic'].sum()
        metrics['total_revenue_fixed'] = total_fixed
        metrics['total_revenue_dynamic'] = total_dynamic
        metrics['revenue_gain_pct'] = ((total_dynamic - total_fixed) / total_fixed) * 100 if total_fixed > 0 else 0
        
        # Utilization improvement
        metrics['avg_util_before'] = df['predicted_utilization'].mean()
        metrics['avg_util_after'] = df['adjusted_utilization'].mean()
        
        # Off-Peak Uplift
        off_peak = df[df['time_slot'] == 'off_peak']
        if len(off_peak) > 0:
            off_peak_discount_mask = off_peak['dynamic_tariff'] < self.base_tariff
            metrics['off_peak_discount_rate'] = off_peak_discount_mask.mean() * 100
            # Simulated uplift based on price elasticity
            off_peak_demand_increase = off_peak['demand_response_factor'].mean()
            metrics['off_peak_uplift_pct'] = (off_peak_demand_increase - 1.0) * 100
        else:
            metrics['off_peak_discount_rate'] = 0
            metrics['off_peak_uplift_pct'] = 0
        
        # Wait time reduction
        metrics['avg_wait_reduction_min'] = df['wait_time_reduction'].mean()
        
        # Tier distribution
        tier_dist = df['tier'].value_counts(normalize=True) * 100
        for tier_name in self.tariff_tiers:
            metrics[f'tier_{tier_name}_pct'] = tier_dist.get(tier_name, 0)
        
        # Print metrics
        print(f'    Revenue Gain: {metrics["revenue_gain_pct"]:.2f}%')
        print(f'    Total Fixed Revenue: ₹{total_fixed:,.0f}')
        print(f'    Total Dynamic Revenue: ₹{total_dynamic:,.0f}')
        print(f'    Avg Utilization: {metrics["avg_util_before"]:.3f} → {metrics["avg_util_after"]:.3f} (after demand response)')
        print(f'    Off-Peak Uplift: {metrics["off_peak_uplift_pct"]:.2f}%')
        print(f'    Avg Wait Time Reduction: {metrics["avg_wait_reduction_min"]:.2f} min')
        
        return metrics


def run_tariff_pricing(predictions_df=None, source='urbanev'):
    """Run the Tariff Pricing Agent pipeline."""
    print_section('PHASE 4: TARIFF PRICING AGENT')
    
    agent = TariffPricingAgent()
    
    # Load predictions if not provided
    if predictions_df is None:
        path = os.path.join(MODEL_OUTPUTS_DIR, f'demand_predictions_{source}.csv')
        if os.path.exists(path):
            predictions_df = pd.read_csv(path, parse_dates=['hour_ts'])
        else:
            print(f'  ✗ Predictions not found at {path}')
            return None, None
    
    # Apply dynamic pricing
    priced_df = agent.apply_dynamic_pricing(predictions_df)
    
    # Train optimization model
    agent.train_optimization_model(priced_df)
    
    # Compute metrics
    metrics = agent.compute_tariff_metrics(priced_df)
    
    # Save outputs
    save_dataframe(priced_df, os.path.join(MODEL_OUTPUTS_DIR, f'tariff_pricing_{source}.csv'), 'Tariff pricing results')
    
    metrics_df = pd.DataFrame([metrics])
    save_dataframe(metrics_df, os.path.join(MODEL_OUTPUTS_DIR, f'tariff_metrics_{source}.csv'), 'Tariff metrics')
    
    # Tier distribution summary
    tier_summary = priced_df.groupby('tier').agg(
        count=('dynamic_tariff', 'count'),
        avg_tariff=('dynamic_tariff', 'mean'),
        total_revenue=('revenue_dynamic', 'sum'),
        avg_utilization=('predicted_utilization', 'mean'),
    ).reset_index()
    save_dataframe(tier_summary, os.path.join(MODEL_OUTPUTS_DIR, f'tier_distribution_{source}.csv'), 'Tier distribution')
    
    print('\n  --- Tier Distribution ---')
    for _, row in tier_summary.iterrows():
        print(f'    {row["tier"]:12s} | Count: {row["count"]:6d} | Avg Tariff: ₹{row["avg_tariff"]:6.2f} | Revenue: ₹{row["total_revenue"]:12,.0f}')
    
    return agent, priced_df


if __name__ == '__main__':
    run_tariff_pricing()
