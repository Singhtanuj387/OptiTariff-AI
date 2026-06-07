"""Shared utility functions."""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import warnings
warnings.filterwarnings('ignore')

def parse_rfc2822_timestamp(ts_str: str) -> pd.Timestamp:
    """Parse RFC 2822 timestamp like 'Thu, 10 Oct 2019 15:48:04 GMT' to pandas Timestamp."""
    if ts_str is None or pd.isna(ts_str):
        return pd.NaT
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(ts_str)
        return pd.Timestamp(dt)
    except Exception:
        return pd.NaT

def get_time_slot(hour: int) -> str:
    """Classify hour into peak/shoulder/off_peak."""
    if hour in list(range(9, 12)) + list(range(18, 21)):
        return 'peak'
    elif hour in list(range(7, 9)) + list(range(12, 18)):
        return 'shoulder'
    else:
        return 'off_peak'

def safe_divide(a, b, default=0.0):
    """Safe division avoiding ZeroDivisionError."""
    if b == 0 or pd.isna(b):
        return default
    return a / b

def create_output_dirs():
    """Create all output directories."""
    from src.config import OUTPUT_DIR, PROCESSED_DATA_DIR, EDA_PLOTS_DIR, MODEL_OUTPUTS_DIR, EVALUATION_DIR, PRESENTATION_DIR
    for d in [OUTPUT_DIR, PROCESSED_DATA_DIR, EDA_PLOTS_DIR, MODEL_OUTPUTS_DIR, EVALUATION_DIR, PRESENTATION_DIR]:
        os.makedirs(d, exist_ok=True)

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def print_step(step: str):
    """Print a formatted step."""
    print(f"  -> {step}")

def print_metric(name: str, value, fmt='.4f'):
    """Print a formatted metric."""
    if isinstance(value, float):
        print(f"    {name}: {value:{fmt}}")
    else:
        print(f"    {name}: {value}")

def save_dataframe(df: pd.DataFrame, filepath: str, description: str = ''):
    """Save DataFrame to CSV with logging."""
    df.to_csv(filepath, index=False)
    print(f"  * Saved {description}: {os.path.basename(filepath)} ({len(df)} rows)")
