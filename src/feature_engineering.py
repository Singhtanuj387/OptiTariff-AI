"""Feature Engineering Module."""

import pandas as pd
import numpy as np
import os
from src.config import *
from src.utils import print_section, print_step, print_metric, save_dataframe


class FeatureEngineer:
    """Engineers features for Demand Prediction and Tariff agents."""
    
    def engineer_acn_features(self, sessions_df: pd.DataFrame, hourly_df: pd.DataFrame) -> pd.DataFrame:
        """Create features for ACN data."""
        print_step('Engineering ACN features...')
        df = hourly_df.copy()
        
        # Station capacity estimation (max concurrent sessions seen)
        station_capacity = df.groupby('station_id')['num_sessions'].max().to_dict()
        df['capacity'] = df['station_id'].map(station_capacity).clip(lower=1)
        
        # Utilization rate
        df['utilization_rate'] = df['num_sessions'] / df['capacity']
        df['utilization_rate'] = df['utilization_rate'].clip(0, 1)
        
        # Cyclical temporal encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7.0)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7.0)
        
        # One-hot encode time slot
        for slot in ['peak', 'shoulder', 'off_peak']:
            df[f'is_{slot}'] = (df['time_slot'] == slot).astype(int)
        
        # Rolling/Lag features
        df = df.sort_values(['station_id', 'hour_ts'])
        df['util_lag1'] = df.groupby('station_id')['utilization_rate'].shift(1)
        df['util_lag2'] = df.groupby('station_id')['utilization_rate'].shift(2)
        df['util_lag24'] = df.groupby('station_id')['utilization_rate'].shift(24)
        df['kwh_lag1'] = df.groupby('station_id')['total_kwh'].shift(1)
        df['kwh_lag24'] = df.groupby('station_id')['total_kwh'].shift(24)
        
        df['util_roll_mean_3h'] = df.groupby('station_id')['utilization_rate'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
        
        return df.dropna()

    def engineer_urbanev_features(self, hourly_df: pd.DataFrame) -> pd.DataFrame:
        """Create features for UrbanEV data."""
        print_step('Engineering UrbanEV features...')
        df = hourly_df.copy()
        
        # Cyclical temporal encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7.0)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7.0)
        
        # One-hot encode time slot
        for slot in ['peak', 'shoulder', 'off_peak']:
            df[f'is_{slot}'] = (df['time_slot'] == slot).astype(int)
            
        # Station characteristics
        df['fast_ratio'] = df['fast_count'] / df['zone_count'].clip(lower=1)
        df['station_size'] = np.where(df['zone_count'] > 20, 'large', np.where(df['zone_count'] > 10, 'medium', 'small'))
        
        # Rolling/Lag features
        df = df.sort_values(['zone_id', 'hour_ts'])
        df['util_lag1'] = df.groupby('zone_id')['avg_utilization'].shift(1)
        df['util_lag2'] = df.groupby('zone_id')['avg_utilization'].shift(2)
        df['util_lag24'] = df.groupby('zone_id')['avg_utilization'].shift(24)
        df['price_lag1'] = df.groupby('zone_id')['avg_price_ratio'].shift(1)
        
        df['util_roll_mean_3h'] = df.groupby('zone_id')['avg_utilization'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
        df['util_roll_std_3h'] = df.groupby('zone_id')['avg_utilization'].rolling(3, min_periods=1).std().reset_index(0, drop=True).fillna(0)
        
        # Target variable standardization
        if 'avg_utilization' in df.columns:
            df['utilization_rate'] = df['avg_utilization']
            
        return df.dropna()

    def get_ml_features(self, source: str) -> list:
        base_features = [
            'hour', 'day_of_week', 'is_weekend',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'is_peak', 'is_shoulder', 'is_off_peak',
            'util_lag1', 'util_lag2', 'util_lag24',
            'util_roll_mean_3h'
        ]
        
        if source == 'urbanev':
            return base_features + ['avg_price_ratio', 'price_lag1', 'fast_ratio', 'is_cbd', 'util_roll_std_3h']
        else:
            return base_features + ['kwh_lag1', 'kwh_lag24']

def run_feature_engineering(acn_sessions=None, acn_hourly=None, uev_hourly=None):
    print_section('PHASE 1B: FEATURE ENGINEERING')
    
    fe = FeatureEngineer()
    
    if acn_sessions is None or acn_hourly is None:
        try:
            acn_sessions = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'acn_sessions.csv'), parse_dates=['connection_time', 'disconnect_time', 'done_charging_time'])
            acn_hourly = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'acn_hourly.csv'), parse_dates=['hour_ts'])
        except: pass
        
    if uev_hourly is None:
        try:
            uev_hourly = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'urbanev_hourly.csv'), parse_dates=['hour_ts'])
        except: pass
    
    acn_featured = None
    if acn_hourly is not None:
        acn_featured = fe.engineer_acn_features(acn_sessions, acn_hourly)
        save_dataframe(acn_featured, os.path.join(PROCESSED_DATA_DIR, 'acn_featured.csv'), 'ACN features')
        
    uev_featured = None
    if uev_hourly is not None:
        uev_featured = fe.engineer_urbanev_features(uev_hourly)
        save_dataframe(uev_featured, os.path.join(PROCESSED_DATA_DIR, 'urbanev_featured.csv'), 'UrbanEV features')
        
    return fe, acn_featured, uev_featured

if __name__ == '__main__':
    run_feature_engineering()
