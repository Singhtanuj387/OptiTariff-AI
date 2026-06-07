"""Global Configuration and Constants."""

import os
import torch

# ─────────────────────────────────────────────
# Global Settings
# ─────────────────────────────────────────────
RANDOM_STATE = 42
TRAIN_TEST_SPLIT = 0.8
NUM_EPISODES = 10
FIGURE_DPI = 300

# ─────────────────────────────────────────────
# Device Configuration
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Config] Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"[Config] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Config] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─────────────────────────────────────────────
# Dataset Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'datasets')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

PROCESSED_DATA_DIR = os.path.join(OUTPUT_DIR, 'processed')
EDA_PLOTS_DIR = os.path.join(OUTPUT_DIR, 'eda')
MODEL_OUTPUTS_DIR = os.path.join(OUTPUT_DIR, 'models')
EVALUATION_DIR = os.path.join(OUTPUT_DIR, 'evaluation')
PRESENTATION_DIR = os.path.join(OUTPUT_DIR, 'presentation')

ACN_DATA_PATH = os.path.join(BASE_DIR, 'acndata_sessions.json')

URBANEV_FILES = {
    'time': os.path.join(DATA_DIR, 'time.csv'),
    'occupancy': os.path.join(DATA_DIR, 'occupancy.csv'),
    'volume': os.path.join(DATA_DIR, 'volume.csv'),
    'duration': os.path.join(DATA_DIR, 'duration.csv'),
    'price': os.path.join(DATA_DIR, 'price.csv'),
    'stations': os.path.join(DATA_DIR, 'stations.csv'),
    'information': os.path.join(DATA_DIR, 'information.csv'),
}

# ─────────────────────────────────────────────
# Agentic AI Pricing Strategy
# ─────────────────────────────────────────────
BASELINE_TARIFF = 15.0  # INR per kWh

TARIFF_TIERS = {
    'low': {'min_util': 0.0, 'max_util': 0.3, 'multiplier': 0.7, 'label': 'Discounted'},
    'normal': {'min_util': 0.3, 'max_util': 0.6, 'multiplier': 1.0, 'label': 'Standard'},
    'high': {'min_util': 0.6, 'max_util': 0.8, 'multiplier': 1.2, 'label': 'Elevated'},
    'surge': {'min_util': 0.8, 'max_util': 0.95, 'multiplier': 1.5, 'label': 'Surge'},
    'critical': {'min_util': 0.95, 'max_util': 1.0, 'multiplier': 2.0, 'label': 'Critical Peak'}
}

TOD_MODIFIERS = {
    'off_peak': 0.9,
    'shoulder': 1.0,
    'peak': 1.15
}

SURGE_THRESHOLD = 0.80
DISCOUNT_THRESHOLD = 0.30
PRICE_ELASTICITY = -0.3

# ─────────────────────────────────────────────
# ML Hyperparameters
# ─────────────────────────────────────────────
XGBOOST_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'tree_method': 'hist'
}

LSTM_PARAMS = {
    'sequence_length': 24,
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'batch_size': 64,
    'epochs': 50,
    'learning_rate': 0.001
}

LEARNING_RATE_AGENT = 0.05

# ─────────────────────────────────────────────
# Visualization Assets
# ─────────────────────────────────────────────
COLOR_PALETTE = {
    'primary': '#0F172A',
    'secondary': '#3B82F6',
    'accent': '#06B6D4',
    'success': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444',
    'off_peak': '#10B981',
    'shoulder': '#F59E0B',
    'peak': '#EF4444'
}
