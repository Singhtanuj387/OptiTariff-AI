"""Monitoring & Learning Agent - Evaluates pricing decisions and learns from outcomes.

Implements a feedback loop that systematically evaluates each pricing decision
against operational outcomes and adjusts pricing parameters.
"""

import pandas as pd
import numpy as np
import os
from src.config import *
from src.utils import print_section, print_step, print_metric, save_dataframe


class MonitoringLearningAgent:
    """Evaluates pricing decisions and continuously improves the pricing strategy.
    
    Feedback Loop:
    1. Collect: Record tariff decisions and simulated outcomes
    2. Evaluate: Compare predicted vs actual performance
    3. Adjust: Update pricing parameters based on feedback
    4. Update: Store learning for next episode
    """
    
    def __init__(self):
        self.episode_history = []
        self.parameter_history = []
        self.learning_rate = LEARNING_RATE_AGENT
        self.current_params = {
            'surge_multiplier': TARIFF_TIERS['surge']['multiplier'],
            'critical_multiplier': TARIFF_TIERS['critical']['multiplier'],
            'discount_multiplier': TARIFF_TIERS['low']['multiplier'],
            'low_multiplier': TARIFF_TIERS['low']['multiplier'],
            'surge_threshold': SURGE_THRESHOLD,
            'discount_threshold': DISCOUNT_THRESHOLD,
            'elasticity': PRICE_ELASTICITY,
        }
        self.parameter_history.append(self.current_params.copy())
    
    def split_episodes(self, df: pd.DataFrame, n_episodes: int = NUM_EPISODES) -> list:
        """Split data into sequential episodes for evaluation."""
        print_step(f'Splitting data into {n_episodes} episodes...')
        
        df = df.sort_values('hour_ts').reset_index(drop=True)
        episode_size = len(df) // n_episodes
        
        episodes = []
        for i in range(n_episodes):
            start = i * episode_size
            end = start + episode_size if i < n_episodes - 1 else len(df)
            episodes.append(df.iloc[start:end].copy())
            print(f'    Episode {i+1}: {len(episodes[-1])} records ({episodes[-1]["hour_ts"].min()} → {episodes[-1]["hour_ts"].max()})')
        
        return episodes
    
    def simulate_outcome(self, episode_df: pd.DataFrame) -> pd.DataFrame:
        """Simulate real-world outcomes given pricing decisions.
        
        Adds noise and applies price elasticity to simulate actual behavior.
        """
        df = episode_df.copy()
        
        # Simulate actual utilization (predicted + noise + demand response)
        noise = np.random.normal(0, 0.05, len(df))
        price_change = (df['dynamic_tariff'] - BASELINE_TARIFF) / BASELINE_TARIFF
        demand_response = price_change * self.current_params['elasticity']
        
        df['actual_utilization'] = (df['predicted_utilization'] + noise + demand_response).clip(0, 1)
        
        # Simulate actual revenue
        energy_col = 'total_volume_kwh' if 'total_volume_kwh' in df.columns else 'total_kwh'
        if energy_col in df.columns:
            base_energy = df[energy_col]
        else:
            base_energy = df.get('expected_load_kwh', 10)
        
        # Actual energy consumed adjusts with utilization change
        df['actual_energy_kwh'] = base_energy * (df['actual_utilization'] / df['predicted_utilization'].clip(lower=0.01)).clip(0.5, 2.0)
        df['actual_revenue'] = df['actual_energy_kwh'] * df['dynamic_tariff']
        df['baseline_revenue'] = df['actual_energy_kwh'] * BASELINE_TARIFF
        
        # Simulated wait time (queue-based)
        df['actual_wait_time'] = (df['actual_utilization'] - 0.85).clip(lower=0) * 20  # minutes
        df['baseline_wait_time'] = (df['predicted_utilization'] - 0.85).clip(lower=0) * 20
        
        return df
    
    def evaluate_episode(self, episode_df: pd.DataFrame, episode_num: int) -> dict:
        """Evaluate a single episode's pricing performance."""
        df = episode_df.copy()
        
        metrics = {
            'episode': episode_num,
            # Revenue
            'total_revenue': df['actual_revenue'].sum(),
            'baseline_revenue': df['baseline_revenue'].sum(),
            'revenue_gain_pct': ((df['actual_revenue'].sum() - df['baseline_revenue'].sum()) / 
                                df['baseline_revenue'].sum() * 100) if df['baseline_revenue'].sum() > 0 else 0,
            # Utilization
            'avg_utilization': df['actual_utilization'].mean(),
            'congestion_rate': (df['actual_utilization'] >= 0.8).mean() * 100,
            'underutil_rate': (df['actual_utilization'] <= 0.3).mean() * 100,
            # Wait time
            'avg_wait_time': df['actual_wait_time'].mean(),
            'baseline_avg_wait': df['baseline_wait_time'].mean(),
            'wait_time_reduction_pct': ((df['baseline_wait_time'].mean() - df['actual_wait_time'].mean()) / 
                                       df['baseline_wait_time'].mean() * 100) if df['baseline_wait_time'].mean() > 0 else 0,
            # Pricing efficiency
            'revenue_per_kwh': (df['actual_revenue'].sum() / df['actual_energy_kwh'].sum()) if df['actual_energy_kwh'].sum() > 0 else 0,
            'avg_dynamic_tariff': df['dynamic_tariff'].mean(),
            # Demand elasticity (observed)
            'demand_elasticity_proxy': np.corrcoef(
                df['dynamic_tariff'].values, df['actual_utilization'].values
            )[0, 1] if len(df) > 2 else 0,
            # Parameters at this episode
            **{f'param_{k}': v for k, v in self.current_params.items()},
        }
        
        return metrics
    
    def learn_from_episode(self, episode_metrics: dict, episode_df: pd.DataFrame):
        """Adjust parameters based on episode outcomes."""
        
        # Learning rules:
        
        # 1. If congestion still high (>15%), increase surge multiplier
        if episode_metrics['congestion_rate'] > 15:
            adjustment = self.learning_rate * 0.1
            self.current_params['surge_multiplier'] = min(2.5, self.current_params['surge_multiplier'] + adjustment)
            self.current_params['critical_multiplier'] = min(3.0, self.current_params['critical_multiplier'] + adjustment)
        elif episode_metrics['congestion_rate'] < 5:
            # Reduce surge if congestion is very low (over-correction)
            adjustment = self.learning_rate * 0.05
            self.current_params['surge_multiplier'] = max(1.2, self.current_params['surge_multiplier'] - adjustment)
        
        # 2. If underutilization high (>40%), increase discount aggressiveness
        if episode_metrics['underutil_rate'] > 40:
            adjustment = self.learning_rate * 0.05
            self.current_params['discount_multiplier'] = max(0.4, self.current_params['discount_multiplier'] - adjustment)
        elif episode_metrics['underutil_rate'] < 20:
            adjustment = self.learning_rate * 0.02
            self.current_params['discount_multiplier'] = min(0.85, self.current_params['discount_multiplier'] + adjustment)
        
        # 3. Update elasticity estimate based on observed correlation
        observed_elasticity = episode_metrics['demand_elasticity_proxy']
        if not np.isnan(observed_elasticity):
            self.current_params['elasticity'] = (
                (1 - self.learning_rate) * self.current_params['elasticity'] +
                self.learning_rate * observed_elasticity * -0.5
            )
        
        self.parameter_history.append(self.current_params.copy())
    
    def apply_updated_pricing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Re-apply pricing with updated parameters."""
        df = df.copy()
        
        conditions = [
            df['predicted_utilization'] >= 0.9,
            df['predicted_utilization'] >= self.current_params['surge_threshold'],
            df['predicted_utilization'] >= 0.5,
            df['predicted_utilization'] >= self.current_params['discount_threshold'],
        ]
        choices = [
            self.current_params['critical_multiplier'],
            self.current_params['surge_multiplier'],
            1.0,  # normal
            self.current_params['low_multiplier'],
        ]
        
        multiplier = np.select(conditions, choices, default=self.current_params['discount_multiplier'])
        
        # Apply time-of-day modifier
        tod_map = {'peak': 1.15, 'shoulder': 1.0, 'off_peak': 0.9}
        tod_mult = df['time_slot'].map(tod_map).fillna(1.0)
        
        df['dynamic_tariff'] = (BASELINE_TARIFF * multiplier * tod_mult).clip(5, 45)
        
        return df
    
    def run_feedback_loop(self, priced_df: pd.DataFrame) -> pd.DataFrame:
        """Run the complete monitoring and learning feedback loop."""
        print_step('Running feedback loop across episodes...')
        
        episodes = self.split_episodes(priced_df)
        
        all_metrics = []
        
        for i, episode in enumerate(episodes):
            episode_num = i + 1
            print(f'\n    --- Episode {episode_num}/{len(episodes)} ---')
            
            # Apply current pricing parameters
            if i > 0:  # After first episode, use updated params
                episode = self.apply_updated_pricing(episode)
            
            # Simulate outcomes
            episode_with_outcomes = self.simulate_outcome(episode)
            
            # Evaluate
            metrics = self.evaluate_episode(episode_with_outcomes, episode_num)
            all_metrics.append(metrics)
            
            print(f'    Revenue: ₹{metrics["total_revenue"]:,.0f} (Gain: {metrics["revenue_gain_pct"]:+.1f}%)')
            print(f'    Avg Util: {metrics["avg_utilization"]:.3f} | Congestion: {metrics["congestion_rate"]:.1f}%')
            print(f'    Wait Time: {metrics["avg_wait_time"]:.2f} min (Reduction: {metrics["wait_time_reduction_pct"]:+.1f}%)')
            print(f'    Pricing Efficiency: ₹{metrics["revenue_per_kwh"]:.2f}/kWh')
            
            # Learn and adjust
            self.learn_from_episode(metrics, episode_with_outcomes)
            
            self.episode_history.append(metrics)
        
        results_df = pd.DataFrame(all_metrics)
        
        # Print learning summary
        print('\n  --- Learning Summary ---')
        first = all_metrics[0]
        last = all_metrics[-1]
        print(f'    Revenue Gain: {first["revenue_gain_pct"]:+.1f}% → {last["revenue_gain_pct"]:+.1f}%')
        print(f'    Congestion Rate: {first["congestion_rate"]:.1f}% → {last["congestion_rate"]:.1f}%')
        print(f'    Pricing Efficiency: ₹{first["revenue_per_kwh"]:.2f} → ₹{last["revenue_per_kwh"]:.2f}/kWh')
        print(f'    Wait Time: {first["avg_wait_time"]:.2f} → {last["avg_wait_time"]:.2f} min')
        
        return results_df
    
    def get_parameter_evolution(self) -> pd.DataFrame:
        """Get the evolution of pricing parameters across episodes."""
        return pd.DataFrame(self.parameter_history)


def run_monitoring(priced_df=None, source='urbanev'):
    """Run the Monitoring & Learning Agent pipeline."""
    print_section('PHASE 5: MONITORING & LEARNING AGENT')
    
    agent = MonitoringLearningAgent()
    
    # Load data if not provided
    if priced_df is None:
        path = os.path.join(MODEL_OUTPUTS_DIR, f'tariff_pricing_{source}.csv')
        if os.path.exists(path):
            priced_df = pd.read_csv(path, parse_dates=['hour_ts'])
        else:
            print(f'  ✗ Tariff pricing data not found at {path}')
            return None, None
    
    # Run feedback loop
    episode_results = agent.run_feedback_loop(priced_df)
    
    # Save results
    save_dataframe(episode_results, os.path.join(MODEL_OUTPUTS_DIR, f'monitoring_episodes_{source}.csv'), 'Monitoring episodes')
    
    # Save parameter evolution
    param_evolution = agent.get_parameter_evolution()
    save_dataframe(param_evolution, os.path.join(MODEL_OUTPUTS_DIR, f'parameter_evolution_{source}.csv'), 'Parameter evolution')
    
    return agent, episode_results


if __name__ == '__main__':
    run_monitoring()
