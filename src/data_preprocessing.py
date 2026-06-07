"""Data Preprocessing Module - ACN-Data & UrbanEV Dataset Processing."""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from src.config import *
from src.utils import *


class ACNDataProcessor:
    """Process ACN-Data (Caltech) JSON charging sessions."""
    
    def __init__(self):
        self.raw_data = None
        self.sessions_df = None
        self.hourly_df = None
    
    def load_json(self) -> pd.DataFrame:
        """Load and parse ACN JSON data into DataFrame."""
        print_step('Loading ACN-Data JSON...')
        with open(ACN_DATA_PATH, 'r') as f:
            data = json.load(f)
        
        meta = data['_meta']
        print(f"    Site: {meta.get('site', 'unknown')}")
        print(f"    Date range: {meta.get('start', 'N/A')} -> {meta.get('end', 'N/A')}")
        
        items = data['_items']
        print(f"    Total sessions in JSON: {len(items)}")
        
        # Extract session data
        records = []
        for item in items:
            record = {
                'session_id': item.get('sessionID', ''),
                'station_id': item.get('stationID', ''),
                'site_id': item.get('siteID', ''),
                'cluster_id': item.get('clusterID', ''),
                'space_id': item.get('spaceID', ''),
                'user_id': item.get('userID', 'anonymous'),
                'connection_time': parse_rfc2822_timestamp(item.get('connectionTime')),
                'disconnect_time': parse_rfc2822_timestamp(item.get('disconnectTime')),
                'done_charging_time': parse_rfc2822_timestamp(item.get('doneChargingTime')),
                'kwh_delivered': item.get('kWhDelivered', 0.0),
                'timezone': item.get('timezone', 'America/Los_Angeles'),
            }
            
            # Extract first user input if available
            user_inputs = item.get('userInputs')
            if user_inputs and len(user_inputs) > 0:
                ui = user_inputs[0]  # Take first input
                record['kwh_requested'] = ui.get('kWhRequested', np.nan)
                record['miles_requested'] = ui.get('milesRequested', np.nan)
                record['minutes_available'] = ui.get('minutesAvailable', np.nan)
                record['wh_per_mile'] = ui.get('WhPerMile', np.nan)
                record['payment_required'] = ui.get('paymentRequired', False)
            else:
                record['kwh_requested'] = np.nan
                record['miles_requested'] = np.nan
                record['minutes_available'] = np.nan
                record['wh_per_mile'] = np.nan
                record['payment_required'] = False
            
            records.append(record)
        
        self.sessions_df = pd.DataFrame(records)
        print(f"    Parsed {len(self.sessions_df)} sessions")
        return self.sessions_df
    
    def clean(self) -> pd.DataFrame:
        """Clean ACN session data."""
        print_step('Cleaning ACN data...')
        df = self.sessions_df.copy()
        initial_count = len(df)
        
        # Fill missing user IDs
        df['user_id'] = df['user_id'].fillna('anonymous')
        
        # Remove sessions with no energy delivered or negative
        df = df[df['kwh_delivered'] > 0].copy()
        print(f"    Removed {initial_count - len(df)} sessions with kWh <= 0")
        
        # Calculate durations in minutes
        df['session_duration_min'] = (df['disconnect_time'] - df['connection_time']).dt.total_seconds() / 60.0
        df['charging_duration_min'] = (df['done_charging_time'] - df['connection_time']).dt.total_seconds() / 60.0
        df['idle_duration_min'] = df['session_duration_min'] - df['charging_duration_min']
        
        # Remove sessions shorter than 1 minute or longer than 48 hours
        df = df[(df['session_duration_min'] > 1) & (df['session_duration_min'] < 2880)].copy()
        
        # Fix negative idle durations
        df['idle_duration_min'] = df['idle_duration_min'].clip(lower=0)
        
        # Fill missing kwh_requested with median
        median_kwh = df['kwh_requested'].median()
        df['kwh_requested'] = df['kwh_requested'].fillna(median_kwh)
        df['minutes_available'] = df['minutes_available'].fillna(df['minutes_available'].median())
        df['wh_per_mile'] = df['wh_per_mile'].fillna(df['wh_per_mile'].median())
        
        # Extract temporal features
        df['hour'] = df['connection_time'].dt.hour
        df['day_of_week'] = df['connection_time'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['date'] = df['connection_time'].dt.date
        df['month'] = df['connection_time'].dt.month
        df['time_slot'] = df['hour'].apply(get_time_slot)
        
        # Charging efficiency
        df['charging_efficiency'] = df['kwh_delivered'] / df['kwh_requested'].clip(lower=0.1)
        df['charging_efficiency'] = df['charging_efficiency'].clip(upper=2.0)  # cap outliers
        
        self.sessions_df = df
        print(f"    Clean dataset: {len(df)} sessions")
        print(f"    Date range: {df['connection_time'].min()} -> {df['connection_time'].max()}")
        print(f"    Unique stations: {df['station_id'].nunique()}")
        print(f"    Missing values: {df.isnull().sum().sum()}")
        return df
    
    def aggregate_hourly(self) -> pd.DataFrame:
        """Aggregate sessions to hourly granularity per station."""
        print_step('Aggregating ACN to hourly granularity...')
        df = self.sessions_df.copy()
        
        # Create hourly floor timestamp (fixed 'h' instead of 'H')
        df['hour_ts'] = df['connection_time'].dt.floor('h')
        
        # Group by station and hour
        hourly = df.groupby(['station_id', 'hour_ts']).agg(
            num_sessions=('session_id', 'count'),
            total_kwh=('kwh_delivered', 'sum'),
            avg_kwh=('kwh_delivered', 'mean'),
            avg_session_duration=('session_duration_min', 'mean'),
            avg_charging_duration=('charging_duration_min', 'mean'),
            avg_idle_duration=('idle_duration_min', 'mean'),
            max_kwh=('kwh_delivered', 'max'),
        ).reset_index()
        
        # Add temporal features
        hourly['hour'] = hourly['hour_ts'].dt.hour
        hourly['day_of_week'] = hourly['hour_ts'].dt.dayofweek
        hourly['is_weekend'] = hourly['day_of_week'].isin([5, 6]).astype(int)
        hourly['time_slot'] = hourly['hour'].apply(get_time_slot)
        hourly['source'] = 'ACN'
        
        self.hourly_df = hourly
        print(f"    Hourly records: {len(hourly)}")
        return hourly


class UrbanEVProcessor:
    """Process UrbanEV (ST-EVCDP) Shenzhen dataset."""
    
    def __init__(self):
        self.time_df = None
        self.occupancy_df = None
        self.volume_df = None
        self.duration_df = None
        self.price_df = None
        self.stations_df = None
        self.info_df = None
        self.hourly_df = None
    
    def load_all(self):
        """Load all UrbanEV CSV files."""
        print_step('Loading UrbanEV datasets...')
        
        # Time index
        self.time_df = pd.read_csv(URBANEV_FILES['time'])
        self.time_df['datetime'] = pd.to_datetime(
            self.time_df[['year', 'month', 'day', 'hour', 'minute', 'second']]
        )
        self.time_df['timestamp_idx'] = range(1, len(self.time_df) + 1)
        print(f"    Time index: {len(self.time_df)} steps, {self.time_df['datetime'].min()} -> {self.time_df['datetime'].max()}")
        
        # Occupancy (wide format: timestamp * station)
        self.occupancy_df = pd.read_csv(URBANEV_FILES['occupancy'])
        print(f"    Occupancy: {self.occupancy_df.shape}")
        
        # Volume
        self.volume_df = pd.read_csv(URBANEV_FILES['volume'])
        print(f"    Volume: {self.volume_df.shape}")
        
        # Duration
        self.duration_df = pd.read_csv(URBANEV_FILES['duration'])
        print(f"    Duration: {self.duration_df.shape}")
        
        # Price
        self.price_df = pd.read_csv(URBANEV_FILES['price'])
        print(f"    Price: {self.price_df.shape}")
        
        # Station info
        self.stations_df = pd.read_csv(URBANEV_FILES['stations'])
        print(f"    Stations: {len(self.stations_df)} stations")
        
        # Grid/zone info
        self.info_df = pd.read_csv(URBANEV_FILES['information'])
        print(f"    Zones: {len(self.info_df)} zones")
    
    def melt_to_long(self, wide_df: pd.DataFrame, value_name: str) -> pd.DataFrame:
        """Convert wide-format (timestamp * station) to long format."""
        station_cols = [c for c in wide_df.columns if c != 'timestamp']
        long_df = wide_df.melt(
            id_vars=['timestamp'],
            value_vars=station_cols,
            var_name='zone_id',
            value_name=value_name
        )
        long_df['zone_id'] = long_df['zone_id'].astype(str)
        return long_df
    
    def process(self) -> pd.DataFrame:
        """Process and merge all UrbanEV data into unified long format."""
        print_step('Processing UrbanEV data...')
        
        # Melt each metric to long format
        occ_long = self.melt_to_long(self.occupancy_df, 'occupancy')
        vol_long = self.melt_to_long(self.volume_df, 'volume_kwh')
        dur_long = self.melt_to_long(self.duration_df, 'duration_hours')
        price_long = self.melt_to_long(self.price_df, 'price_ratio')
        
        # Merge all metrics
        merged = occ_long.copy()
        merged = merged.merge(vol_long, on=['timestamp', 'zone_id'], how='left')
        merged = merged.merge(dur_long, on=['timestamp', 'zone_id'], how='left')
        merged = merged.merge(price_long, on=['timestamp', 'zone_id'], how='left')
        
        # Add datetime from time index
        time_map = dict(zip(self.time_df['timestamp_idx'], self.time_df['datetime']))
        merged['datetime'] = merged['timestamp'].map(time_map)
        
        # Add zone info
        zone_info = self.info_df[['grid', 'count', 'fast_count', 'slow_count', 'area', 'lon', 'la', 'CBD', 'dynamic_pricing']].copy()
        zone_info['grid'] = zone_info['grid'].astype(str)
        merged = merged.merge(zone_info, left_on='zone_id', right_on='grid', how='left')
        
        # Calculate utilization rate
        merged['utilization_rate'] = merged['occupancy'] / merged['count'].clip(lower=1)
        merged['utilization_rate'] = merged['utilization_rate'].clip(0, 1)
        
        print(f"    Merged long format: {len(merged)} rows")
        return merged
    
    def aggregate_hourly(self, long_df: pd.DataFrame) -> pd.DataFrame:
        """Resample from 5-min to hourly per zone."""
        print_step('Aggregating UrbanEV to hourly...')
        df = long_df.copy()
        # fixed 'h' instead of 'H'
        df['hour_ts'] = df['datetime'].dt.floor('h')
        
        hourly = df.groupby(['zone_id', 'hour_ts']).agg(
            avg_occupancy=('occupancy', 'mean'),
            total_volume_kwh=('volume_kwh', 'sum'),
            avg_duration_hours=('duration_hours', 'mean'),
            avg_price_ratio=('price_ratio', 'mean'),
            avg_utilization=('utilization_rate', 'mean'),
            max_utilization=('utilization_rate', 'max'),
            zone_count=('count', 'first'),
            fast_count=('fast_count', 'first'),
            slow_count=('slow_count', 'first'),
            is_cbd=('CBD', 'first')
        ).reset_index()
        
        hourly['hour'] = hourly['hour_ts'].dt.hour
        hourly['day_of_week'] = hourly['hour_ts'].dt.dayofweek
        hourly['is_weekend'] = hourly['day_of_week'].isin([5, 6]).astype(int)
        hourly['time_slot'] = hourly['hour'].apply(get_time_slot)
        hourly['source'] = 'UrbanEV'
        
        self.hourly_df = hourly
        print(f"    Hourly records: {len(hourly)}")
        return hourly

def run_preprocessing():
    print_section('PHASE 1: DATA PREPROCESSING')
    create_output_dirs()
    
    # ACN Data
    acn = ACNDataProcessor()
    acn.load_json()
    acn_sessions = acn.clean()
    acn_hourly = acn.aggregate_hourly()
    save_dataframe(acn_sessions, os.path.join(PROCESSED_DATA_DIR, 'acn_sessions.csv'), 'ACN sessions')
    save_dataframe(acn_hourly, os.path.join(PROCESSED_DATA_DIR, 'acn_hourly.csv'), 'ACN hourly')
    
    # UrbanEV Data
    uev = UrbanEVProcessor()
    uev.load_all()
    uev_long = uev.process()
    uev_hourly = uev.aggregate_hourly(uev_long)
    save_dataframe(uev_hourly, os.path.join(PROCESSED_DATA_DIR, 'urbanev_hourly.csv'), 'UrbanEV hourly')
    
    return acn_sessions, acn_hourly, uev_hourly

if __name__ == '__main__':
    run_preprocessing()
