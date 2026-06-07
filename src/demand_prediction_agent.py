"""Demand Prediction Agent - Predicts future charging demand and station utilization.

Uses XGBoost (GPU-accelerated), PyTorch LSTM, and Random Forest models.
"""

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

from src.config import *
from src.utils import print_section, print_step, print_metric, save_dataframe


# ─────────────────────────────────────────────
# PyTorch LSTM Model
# ─────────────────────────────────────────────
class ChargingDataset(Dataset):
    """PyTorch Dataset for time-series charging data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 24):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len
    
    def __len__(self):
        return max(0, len(self.X) - self.seq_len)
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.seq_len]
        y_val = self.y[idx + self.seq_len - 1]
        return X_seq, y_val


class LSTMDemandModel(nn.Module):
    """LSTM model for demand prediction."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze(-1)


# ─────────────────────────────────────────────
# Demand Prediction Agent
# ─────────────────────────────────────────────
class DemandPredictionAgent:
    """Autonomous agent that predicts future charging demand and station utilization.
    
    Models:
    - XGBoost Regressor (primary, GPU-accelerated): predicts utilization_rate
    - Random Forest Classifier: predicts congestion probability (>80%)
    - LSTM (PyTorch CUDA): sequence-based demand prediction
    - Linear Regression (baseline): for comparison
    """
    
    def __init__(self):
        self.xgb_model = None
        self.rf_classifier = None
        self.lstm_model = None
        self.linear_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.metrics = {}
    
    def perceive(self, df: pd.DataFrame, feature_cols: list, target_col: str = 'utilization_rate'):
        """Ingest data and prepare train/test splits."""
        print_step('Agent perceiving data...')
        
        self.feature_columns = feature_cols
        self.target_col = target_col
        
        # Drop rows with NaN in features or target
        valid_cols = [c for c in feature_cols if c in df.columns]
        self.feature_columns = valid_cols
        
        df_clean = df.dropna(subset=valid_cols + [target_col]).copy()
        print(f'    Clean data: {len(df_clean)} rows, {len(valid_cols)} features')
        
        # Temporal split (80/20)
        split_idx = int(len(df_clean) * TRAIN_TEST_SPLIT)
        
        self.X_train = df_clean[valid_cols].iloc[:split_idx].values
        self.X_test = df_clean[valid_cols].iloc[split_idx:].values
        self.y_train = df_clean[target_col].iloc[:split_idx].values
        self.y_test = df_clean[target_col].iloc[split_idx:].values
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Binary target for congestion classification
        self.y_train_binary = (self.y_train >= SURGE_THRESHOLD).astype(int)
        self.y_test_binary = (self.y_test >= SURGE_THRESHOLD).astype(int)
        
        # Store full test DataFrame for later analysis
        self.test_df = df_clean.iloc[split_idx:].copy()
        
        print(f'    Train: {len(self.X_train)} | Test: {len(self.X_test)}')
        print(f'    Congestion rate (train): {self.y_train_binary.mean():.2%}')
        print(f'    Congestion rate (test): {self.y_test_binary.mean():.2%}')
    
    def train_xgboost(self):
        """Train XGBoost regressor with GPU acceleration."""
        print_step('Training XGBoost Regressor (GPU)...')
        
        params = XGBOOST_PARAMS.copy()
        # Check if CUDA is available for XGBoost
        try:
            self.xgb_model = xgb.XGBRegressor(**params)
            self.xgb_model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_test, self.y_test)],
                verbose=False
            )
            print('    ✓ XGBoost trained with GPU acceleration')
        except Exception as e:
            print(f'    ⚠ GPU failed ({e}), falling back to CPU...')
            params['tree_method'] = 'hist'
            params.pop('device', None)
            self.xgb_model = xgb.XGBRegressor(**params)
            self.xgb_model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_test, self.y_test)],
                verbose=False
            )
            print('    ✓ XGBoost trained on CPU')
        
        # Predictions & metrics
        y_pred = self.xgb_model.predict(self.X_test)
        self.metrics['xgboost'] = self._compute_regression_metrics(self.y_test, y_pred)
        self._print_metrics('XGBoost', self.metrics['xgboost'])
        
        return y_pred
    
    def train_random_forest_classifier(self):
        """Train Random Forest for congestion classification."""
        print_step('Training Random Forest Classifier...')
        
        self.rf_classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.rf_classifier.fit(self.X_train_scaled, self.y_train_binary)
        
        y_pred = self.rf_classifier.predict(self.X_test_scaled)
        y_prob = self.rf_classifier.predict_proba(self.X_test_scaled)[:, 1]
        
        acc = accuracy_score(self.y_test_binary, y_pred)
        print(f'    ✓ Congestion Classification Accuracy: {acc:.4f}')
        self.metrics['rf_classifier'] = {'accuracy': acc, 'predictions': y_prob}
        
        return y_prob
    
    def train_lstm(self):
        """Train LSTM model on CUDA."""
        print_step(f'Training LSTM on {DEVICE}...')
        
        seq_len = LSTM_PARAMS['sequence_length']
        
        if len(self.X_train_scaled) < seq_len + 10:
            print('    ⚠ Not enough data for LSTM, skipping...')
            return None
        
        # Create datasets
        train_dataset = ChargingDataset(self.X_train_scaled, self.y_train, seq_len)
        test_dataset = ChargingDataset(self.X_test_scaled, self.y_test, seq_len)
        
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            print('    ⚠ Not enough sequential data for LSTM, skipping...')
            return None
        
        train_loader = DataLoader(train_dataset, batch_size=LSTM_PARAMS['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=LSTM_PARAMS['batch_size'], shuffle=False)
        
        # Build model
        input_size = self.X_train_scaled.shape[1]
        self.lstm_model = LSTMDemandModel(
            input_size=input_size,
            hidden_size=LSTM_PARAMS['hidden_size'],
            num_layers=LSTM_PARAMS['num_layers'],
            dropout=LSTM_PARAMS['dropout']
        ).to(DEVICE)
        
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=LSTM_PARAMS['learning_rate'])
        criterion = nn.MSELoss()
        
        # Training loop
        self.lstm_model.train()
        best_loss = float('inf')
        for epoch in range(LSTM_PARAMS['epochs']):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                
                optimizer.zero_grad()
                output = self.lstm_model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            if (epoch + 1) % 10 == 0:
                print(f'    Epoch {epoch+1}/{LSTM_PARAMS["epochs"]}: Loss = {avg_loss:.6f}')
            
            if avg_loss < best_loss:
                best_loss = avg_loss
        
        # Evaluation
        self.lstm_model.eval()
        all_preds = []
        all_actuals = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                preds = self.lstm_model(X_batch).cpu().numpy()
                all_preds.extend(preds)
                all_actuals.extend(y_batch.numpy())
        
        all_preds = np.array(all_preds)
        all_actuals = np.array(all_actuals)
        
        self.metrics['lstm'] = self._compute_regression_metrics(all_actuals, all_preds)
        self._print_metrics('LSTM', self.metrics['lstm'])
        
        return all_preds
    
    def train_linear_baseline(self):
        """Train linear regression baseline."""
        print_step('Training Linear Regression (baseline)...')
        
        self.linear_model = LinearRegression()
        self.linear_model.fit(self.X_train_scaled, self.y_train)
        
        y_pred = self.linear_model.predict(self.X_test_scaled)
        self.metrics['linear'] = self._compute_regression_metrics(self.y_test, y_pred)
        self._print_metrics('Linear Regression', self.metrics['linear'])
        
        return y_pred
    
    def predict(self, features: np.ndarray) -> dict:
        """Generate predictions from all models."""
        features_scaled = self.scaler.transform(features.reshape(1, -1) if features.ndim == 1 else features)
        
        result = {}
        if self.xgb_model:
            result['predicted_utilization'] = self.xgb_model.predict(features.reshape(1, -1) if features.ndim == 1 else features)
        if self.rf_classifier:
            result['congestion_probability'] = self.rf_classifier.predict_proba(features_scaled)[:, 1]
        
        return result
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get XGBoost feature importance."""
        if self.xgb_model is None:
            return pd.DataFrame()
        
        importance = self.xgb_model.feature_importances_
        fi_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return fi_df
    
    def get_predictions_df(self) -> pd.DataFrame:
        """Return test predictions as a DataFrame for downstream agents."""
        result = self.test_df.copy()
        
        if self.xgb_model:
            result['predicted_utilization'] = self.xgb_model.predict(self.X_test)
        if self.rf_classifier:
            result['congestion_probability'] = self.rf_classifier.predict_proba(self.X_test_scaled)[:, 1]
        if self.linear_model:
            result['predicted_util_linear'] = self.linear_model.predict(self.X_test_scaled)
        
        # Expected charging load
        if 'total_volume_kwh' in result.columns:
            result['expected_load_kwh'] = result['predicted_utilization'] * result['total_volume_kwh'].mean() * 2
        elif 'total_kwh' in result.columns:
            result['expected_load_kwh'] = result['predicted_utilization'] * result['total_kwh'].mean() * 2
        else:
            result['expected_load_kwh'] = result['predicted_utilization'] * 10  # fallback
        
        return result
    
    def _compute_regression_metrics(self, y_true, y_pred) -> dict:
        """Compute RMSE, MAE, R²."""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def _print_metrics(self, model_name: str, metrics: dict):
        """Print formatted metrics."""
        print(f'    {model_name} Results:')
        print_metric('RMSE', metrics['rmse'])
        print_metric('MAE', metrics['mae'])
        print_metric('R² Score', metrics['r2'])


def run_demand_prediction(featured_df=None, feature_cols=None, source='urbanev'):
    """Run the Demand Prediction Agent pipeline."""
    print_section('PHASE 3: DEMAND PREDICTION AGENT')
    
    agent = DemandPredictionAgent()
    
    # Load data if not provided
    if featured_df is None:
        if source == 'urbanev':
            path = os.path.join(PROCESSED_DATA_DIR, 'urbanev_featured.csv')
        else:
            path = os.path.join(PROCESSED_DATA_DIR, 'acn_featured.csv')
        
        if os.path.exists(path):
            featured_df = pd.read_csv(path, parse_dates=['hour_ts'])
        else:
            print(f'  ✗ Featured data not found at {path}')
            return None
    
    if feature_cols is None:
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        feature_cols = fe.get_ml_features(source)
    
    # Filter to valid features only
    valid_features = [c for c in feature_cols if c in featured_df.columns]
    print(f'  Using {len(valid_features)}/{len(feature_cols)} features')
    
    # Sort by time for proper temporal split
    featured_df = featured_df.sort_values('hour_ts').reset_index(drop=True)
    
    # Perceive data
    agent.perceive(featured_df, valid_features)
    
    # Train all models
    xgb_preds = agent.train_xgboost()
    congestion_probs = agent.train_random_forest_classifier()
    lstm_preds = agent.train_lstm()
    linear_preds = agent.train_linear_baseline()
    
    # Feature importance
    fi_df = agent.get_feature_importance()
    save_dataframe(fi_df, os.path.join(MODEL_OUTPUTS_DIR, f'feature_importance_{source}.csv'), 'Feature importance')
    
    # Save predictions
    predictions_df = agent.get_predictions_df()
    save_dataframe(predictions_df, os.path.join(MODEL_OUTPUTS_DIR, f'demand_predictions_{source}.csv'), 'Demand predictions')
    
    # Save metrics summary
    metrics_rows = []
    for model_name, m in agent.metrics.items():
        if 'rmse' in m:
            metrics_rows.append({'model': model_name, 'rmse': m['rmse'], 'mae': m['mae'], 'r2': m['r2']})
    
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        save_dataframe(metrics_df, os.path.join(MODEL_OUTPUTS_DIR, f'demand_metrics_{source}.csv'), 'Demand prediction metrics')
    
    # Print model comparison
    print('\n  --- Model Comparison ---')
    for model_name, m in agent.metrics.items():
        if 'rmse' in m:
            print(f'    {model_name:20s} | RMSE: {m["rmse"]:.4f} | MAE: {m["mae"]:.4f} | R²: {m["r2"]:.4f}')
    
    return agent, predictions_df


if __name__ == '__main__':
    run_demand_prediction()
