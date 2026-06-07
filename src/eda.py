"""Exploratory Data Analysis Module."""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.config import *
from src.utils import print_section, print_step, save_dataframe

def run_eda(acn_sessions=None, acn_featured=None, uev_featured=None):
    """Run EDA generation."""
    print_section('PHASE 2: EXPLORATORY DATA ANALYSIS')
    
    if uev_featured is None:
        try:
            uev_featured = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'urbanev_featured.csv'), parse_dates=['hour_ts'])
        except: pass
        
    if uev_featured is not None:
        print_step('Generating UrbanEV EDA plots...')
        
        # 1. Average Utilization by Hour
        fig, ax = plt.subplots(figsize=(10, 6))
        hourly_util = uev_featured.groupby('hour')['utilization_rate'].mean()
        ax.plot(hourly_util.index, hourly_util.values, marker='o', color=COLOR_PALETTE['primary'], linewidth=2)
        ax.axhline(y=0.8, color=COLOR_PALETTE['danger'], linestyle='--', label='Surge Threshold')
        ax.set_title('Average Utilization by Hour of Day', fontweight='bold')
        ax.set_xlabel('Hour (0-23)')
        ax.set_ylabel('Utilization Rate')
        ax.set_xticks(range(24))
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_PLOTS_DIR, 'urbanev_hourly_utilization.png'), dpi=FIGURE_DPI)
        plt.close()
        
        # 2. Utilization vs Cost
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(uev_featured['avg_price_ratio'], uev_featured['utilization_rate'], alpha=0.1, color=COLOR_PALETTE['secondary'])
        ax.set_title('Utilization Rate vs Grid Electricity Cost', fontweight='bold')
        ax.set_xlabel('Price Ratio')
        ax.set_ylabel('Utilization Rate')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_PLOTS_DIR, 'urbanev_utilization_vs_cost.png'), dpi=FIGURE_DPI)
        plt.close()
        
        # 3. Congestion Heatmap
        heatmap_data = uev_featured.pivot_table(index='day_of_week', columns='hour', values='utilization_rate', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax)
        ax.set_title('Utilization Heatmap (Day vs Hour)', fontweight='bold')
        ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_PLOTS_DIR, 'urbanev_utilization_heatmap.png'), dpi=FIGURE_DPI)
        plt.close()

if __name__ == '__main__':
    run_eda()
