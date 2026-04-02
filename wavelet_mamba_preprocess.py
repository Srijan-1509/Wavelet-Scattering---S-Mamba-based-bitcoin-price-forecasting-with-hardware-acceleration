"""
Wavelet-Mamba Data Preprocessing Pipeline
==========================================
Reads btc_15m_data_2018_to_2025.csv directly, engineers multi-variate features,
creates sliding windows, and splits 80/20 for training.

Features engineered:
  1. Close price (normalized)
  2. Open price (normalized)
  3. High price (normalized)
  4. Low price (normalized)
  5. Log Volume
  6. Returns (pct change of Close)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
import warnings

warnings.filterwarnings('ignore')


class WaveletMambaProcessor:
    def __init__(self, csv_path='btc_15m_data_2018_to_2025.csv',
                 window_size=64, split_ratio=0.8):
        self.csv_path = csv_path
        self.window_size = window_size  # power of 2 for wavelet compatibility
        self.split_ratio = split_ratio
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def load_and_engineer_features(self):
        """Load CSV and create multi-variate feature matrix"""
        print("=" * 80)
        print("WAVELET-MAMBA DATA PREPROCESSING PIPELINE")
        print("=" * 80)

        print(f"\n[*] Loading: {self.csv_path}")
        df = pd.read_csv(self.csv_path)

        # Parse columns
        print(f"[*] Raw columns: {df.columns.tolist()}")
        print(f"[*] Total rows: {len(df)}")

        # Clean data
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Engineer features
        print("\n[*] Engineering features...")

        # 1-4: OHLC prices (raw, will be scaled later)
        features = pd.DataFrame()
        features['close'] = df['Close'].values
        features['open'] = df['Open'].values
        features['high'] = df['High'].values
        features['low'] = df['Low'].values

        # 5: Log volume (handles large range)
        features['log_volume'] = np.log1p(df['Volume'].values)

        # 6: Returns (percentage change)
        features['returns'] = df['Close'].pct_change().fillna(0).values
        features['returns'] = features['returns'].clip(-0.1, 0.1)

        # 7: RSI (14-period)
        close_series = pd.Series(df['Close'].values)
        delta = close_series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14, min_periods=1).mean()
        loss_val = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss_val + 1e-10)
        features['rsi'] = (100 - 100 / (1 + rs)).fillna(50).values

        # 8: MACD signal (12, 26, 9)
        ema12 = close_series.ewm(span=12, adjust=False).mean()
        ema26 = close_series.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        features['macd'] = (macd_line - macd_signal).fillna(0).values

        # 9: Volatility (rolling 14-period std of returns)
        features['volatility'] = pd.Series(features['returns'].values).rolling(
            window=14, min_periods=1).std().fillna(0).values

        print(f"[OK] Features engineered: {list(features.columns)}")
        print(f"[OK] Feature matrix shape: {features.shape}")

        return features, df['Close'].values

    def create_windows(self, features, close_prices):
        """Create sliding window samples"""
        print(f"\n[*] Creating sliding windows (size={self.window_size})...")

        feature_matrix = features.values  # (N, 6)
        X = []
        y_price = []
        y_direction = []

        for i in range(self.window_size, len(feature_matrix) - 1):
            # Input: window of features
            window = feature_matrix[i - self.window_size: i]
            X.append(window)

            # Target: next Close price
            target_price = close_prices[i + 1]
            current_price = close_prices[i]
            y_price.append(target_price)

            # Direction: up or down
            direction = 1 if target_price > current_price else 0
            y_direction.append(direction)

        X = np.array(X, dtype=np.float32)           # (samples, window, features)
        y_price = np.array(y_price, dtype=np.float32)
        y_direction = np.array(y_direction, dtype=np.int64)

        print(f"[OK] Windows created: {X.shape[0]} samples")
        print(f"     X shape: {X.shape}  (samples, window_size={self.window_size}, features={X.shape[2]})")
        print(f"     y_price shape: {y_price.shape}")

        return X, y_price, y_direction

    def split_and_scale(self, X, y_price, y_direction):
        """Chronological split + scaling"""
        print(f"\n[*] Splitting data (ratio={self.split_ratio})...")

        split_idx = int(len(y_price) * self.split_ratio)

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_price_train, y_price_test = y_price[:split_idx], y_price[split_idx:]
        y_dir_train, y_dir_test = y_direction[:split_idx], y_direction[split_idx:]

        print(f"     Train: {X_train.shape[0]} samples")
        print(f"     Test:  {X_test.shape[0]} samples")

        # Scale features (fit on train only)
        print("[*] Scaling features (StandardScaler)...")
        n_train, w, f = X_train.shape
        n_test = X_test.shape[0]

        # Reshape to 2D for scaling
        X_train_flat = X_train.reshape(-1, f)
        self.feature_scaler.fit(X_train_flat)

        X_train = self.feature_scaler.transform(X_train_flat).reshape(n_train, w, f)
        X_test = self.feature_scaler.transform(X_test.reshape(-1, f)).reshape(n_test, w, f)

        # Scale target
        y_price_train = self.target_scaler.fit_transform(
            y_price_train.reshape(-1, 1)).flatten()
        y_price_test = self.target_scaler.transform(
            y_price_test.reshape(-1, 1)).flatten()

        print("[OK] Scaling complete")

        return (X_train.astype(np.float32), X_test.astype(np.float32),
                y_price_train.astype(np.float32), y_price_test.astype(np.float32),
                y_dir_train, y_dir_test)

    def process(self):
        """Full pipeline"""
        features, close_prices = self.load_and_engineer_features()
        X, y_price, y_direction = self.create_windows(features, close_prices)
        return self.split_and_scale(X, y_price, y_direction)


def main():
    processor = WaveletMambaProcessor()
    result = processor.process()

    (X_train, X_test,
     y_price_train, y_price_test,
     y_dir_train, y_dir_test) = result

    # Save
    print("\n[*] Saving processed data...")
    np.save('wm_X_train.npy', X_train)
    np.save('wm_X_test.npy', X_test)
    np.save('wm_y_price_train.npy', y_price_train)
    np.save('wm_y_price_test.npy', y_price_test)
    np.save('wm_y_dir_train.npy', y_dir_train)
    np.save('wm_y_dir_test.npy', y_dir_test)

    with open('wm_target_scaler.pkl', 'wb') as f:
        pickle.dump(processor.target_scaler, f)

    with open('wm_feature_scaler.pkl', 'wb') as f:
        pickle.dump(processor.feature_scaler, f)

    print("\n" + "=" * 80)
    print("[OK] PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_price_train.shape}")
    print(f"  y_test:  {y_price_test.shape}")
    print(f"\nSaved files:")
    print(f"  wm_X_train.npy, wm_X_test.npy")
    print(f"  wm_y_price_train.npy, wm_y_price_test.npy")
    print(f"  wm_y_dir_train.npy, wm_y_dir_test.npy")
    print(f"  wm_target_scaler.pkl, wm_feature_scaler.pkl")


if __name__ == "__main__":
    main()
