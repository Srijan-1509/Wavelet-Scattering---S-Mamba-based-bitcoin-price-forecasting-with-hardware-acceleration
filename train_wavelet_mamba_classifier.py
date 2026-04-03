import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, accuracy_score,
                             roc_curve, auc, classification_report)
import pickle, os, time, math
from wavelet_scattering import WaveletScatteringNetwork
from mamba_predictor import (SelectiveSSM, CMBlock, CBlock,
                              TemporalAttentionPool, LearnablePositionalEncoding)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ============================================================================
# Model: Wavelet-Mamba Binary Classifier v2 — Gated + Regime-Conditioned
# ============================================================================

class WaveletMambaClassifier(nn.Module):
    """
    Enhanced Wavelet-Mamba for Binary Classification v2.

    Key architectural changes from v1:
      - Volume-weighted wavelet scattering with temporal deltas
      - Regime conditioning in Mamba SSM (ATR-modulated delta)
      - Gated classification head: separate wavelet energy + regime pathways
      - Decoupled LayerNorms for classification vs Mamba paths
    """

    # Volume channel index and regime channel index — updated for relative-only features
    VOLUME_FEAT_IDX = 0   # log_volume is feature index 0
    ATR_FEAT_IDX = 8      # atr_norm is feature index 8

    def __init__(self, n_features=12, window_size=64, d_model=128, d_state=16, dropout=0.25):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features

        # Layer 1: Input Projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Layer 2: Positional Encoding
        self.pos_enc = LearnablePositionalEncoding(d_model, max_len=window_size + 16)

        # Layer 3: Wavelet Scattering (enhanced with volume weighting + deltas)
        self.wavelet_scatter = WaveletScatteringNetwork(
            in_channels=d_model, out_channels=d_model,
            J=3, Q=1, kernel_size=min(16, window_size // 2)
        )

        # Layers 4-6: Three C-Blocks (CryptoMamba-style hierarchy)
        seq1 = max(window_size // 2, 8)
        seq2 = max(window_size // 4, 4)
        self.cblock1 = CBlock(d_model, n_cmblocks=2, d_state=d_state,
                               target_seq_len=seq1, dropout=dropout)
        self.cblock2 = CBlock(d_model, n_cmblocks=2, d_state=d_state,
                               target_seq_len=seq2, dropout=dropout)
        self.cblock3 = CBlock(d_model, n_cmblocks=2, d_state=d_state,
                               target_seq_len=None, dropout=dropout)

        # Layer 7: Merge (with separate LayerNorm for classification path)
        self.merge_proj1 = nn.Linear(d_model, d_model)
        self.merge_proj2 = nn.Linear(d_model, d_model)
        self.merge_proj3 = nn.Linear(d_model, d_model)
        self.merge_norm = nn.LayerNorm(d_model)

        # Layer 8: Temporal Attention Pooling
        self.temporal_pool = TemporalAttentionPool(d_model)

        # ====== GATED CLASSIFICATION HEAD ======

        # Separate LayerNorm for classification pathway
        self.cls_norm = nn.LayerNorm(d_model)

        # Gate mechanism: takes concat(mamba_pooled, wavelet_energy) -> gated features
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # Regime context embedding (ATR -> conditioning bias)
        self.regime_embed = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Deep Classification Head (after gating + regime conditioning)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Binary Output
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        """
        x: (B, T, n_features) — includes volume at idx 4 and ATR at idx 12
        """
        batch, seq_len, _ = x.shape

        # Extract volume weights for wavelet scattering: (B, 1, T)
        vol_weights = x[:, :, self.VOLUME_FEAT_IDX].unsqueeze(1)  # (B, 1, T)
        # Normalize volume weights per sample to [0, 1] range
        vol_min = vol_weights.min(dim=2, keepdim=True).values
        vol_max = vol_weights.max(dim=2, keepdim=True).values
        vol_weights = (vol_weights - vol_min) / (vol_max - vol_min + 1e-8)

        # Extract ATR as regime signal: (B, T, 1)
        regime_signal = x[:, :, self.ATR_FEAT_IDX].unsqueeze(2)  # (B, T, 1)

        # Layer 1: Input projection
        h = self.input_proj(x)  # (B, T, d_model)

        # Layer 2: Positional encoding
        h = self.pos_enc(h)

        # Layer 3: Wavelet scattering (channel-first) with volume weighting
        h_t = h.transpose(1, 2)  # (B, d_model, T)
        h_t, wavelet_energy = self.wavelet_scatter(h_t, volume_weights=vol_weights)
        # wavelet_energy: (B, d_model) — global avg energy for gating
        h = h_t.transpose(1, 2)  # (B, T, d_model)

        # Layers 4-6: C-Blocks with regime conditioning
        out1 = self.cblock1(h, regime_signal=regime_signal)
        out2 = self.cblock2(out1, regime_signal=regime_signal)
        out3 = self.cblock3(out2, regime_signal=regime_signal)

        # Layer 7: Merge
        out1_pooled = F.adaptive_avg_pool1d(
            out1.transpose(1, 2), out3.shape[1]).transpose(1, 2)
        merged = (self.merge_proj1(out1_pooled) +
                  self.merge_proj2(out2) +
                  self.merge_proj3(out3))
        merged = self.merge_norm(merged)

        # Layer 8: Temporal Attention Pooling
        mamba_pooled = self.temporal_pool(merged)  # (B, d_model)

        # ====== GATED CLASSIFICATION HEAD ======

        # Separate normalization for classification
        mamba_cls = self.cls_norm(mamba_pooled)

        # Gate: combine mamba hidden state with wavelet energy
        gate_input = torch.cat([mamba_cls, wavelet_energy], dim=-1)  # (B, 2*d_model)
        gate_weights = self.gate(gate_input)  # (B, d_model) — sigmoid gating
        gated_features = mamba_cls * gate_weights  # element-wise gating

        # Regime context: add ATR-based conditioning bias
        atr_mean = regime_signal.mean(dim=1)  # (B, 1) — avg ATR over window
        regime_bias = self.regime_embed(atr_mean)  # (B, d_model)
        gated_features = gated_features + regime_bias

        # Classification
        features = self.classifier(gated_features)
        logit = self.output(features)
        return logit

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Combined Asymmetric + Differentiable F-beta Loss
# ============================================================================

class CombinedClassificationLoss(nn.Module):
    """
    Combined loss for binary classification:
      1. Asymmetric Focal Loss (per-sample, penalizes FP more)
      2. Differentiable F-beta Loss (global, β=0.5 weights precision > recall)

    Total: L = w_focal * L_asym + w_fbeta * (1 - F_beta_soft)
    """

    def __init__(self, alpha=0.5, gamma_pos=1.0, gamma_neg=4.0,
                 label_smoothing=0.1, beta=0.5,
                 w_focal=0.6, w_fbeta=0.4):
        super().__init__()
        self.alpha = alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.label_smoothing = label_smoothing
        self.beta = beta
        self.w_focal = w_focal
        self.w_fbeta = w_fbeta

    def forward(self, logits, targets):
        # ---- Asymmetric Focal Component ----
        targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')

        p_t = probs * targets + (1 - probs) * (1 - targets)
        gamma = targets * self.gamma_pos + (1 - targets) * self.gamma_neg
        focal_weight = (1 - p_t) ** gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = (alpha_t * focal_weight * bce).mean()

        # ---- Differentiable F-beta Component ----
        # Use soft probabilities for differentiable F-beta
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()

        beta_sq = self.beta ** 2
        precision_soft = tp / (tp + fp + 1e-8)
        recall_soft = tp / (tp + fn + 1e-8)
        fbeta_soft = (1 + beta_sq) * precision_soft * recall_soft / \
                     (beta_sq * precision_soft + recall_soft + 1e-8)
        fbeta_loss = 1.0 - fbeta_soft

        return self.w_focal * focal_loss + self.w_fbeta * fbeta_loss


# ============================================================================
# Data Preprocessing — Enhanced with Bollinger Bands, ATR, Regime Labels
# ============================================================================

def preprocess_data(csv_path='btc_15m_data_2018_to_2025.csv',
                    window_size=64, split_ratio=0.8, horizon=4,
                    threshold=0.001, purge_gap=256):
    print("=" * 80)
    print("BINARY CLASSIFICATION DATA PREPROCESSING v2")
    print("=" * 80)

    df = pd.read_csv(csv_path)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[*] Total rows: {len(df)}")

    # ===== Engineer RELATIVE features only (12 total) =====
    # NO absolute prices — only normalized/relative features for direction prediction
    features = pd.DataFrame()
    close_arr = df['Close'].values
    high_arr = df['High'].values
    low_arr = df['Low'].values
    open_arr = df['Open'].values
    close_series = pd.Series(close_arr)

    # Feature 0: log_volume (relative, scale-invariant)
    features['log_volume'] = np.log1p(df['Volume'].values)

    # Feature 1: returns (pct change)
    features['returns'] = df['Close'].pct_change().fillna(0).clip(-0.1, 0.1).values

    # Feature 2: RSI (bounded 0-100)
    delta_price = close_series.diff()
    gain = delta_price.where(delta_price > 0, 0.0).rolling(14, min_periods=1).mean()
    loss_v = (-delta_price.where(delta_price < 0, 0.0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss_v + 1e-10)
    features['rsi'] = (100 - 100 / (1 + rs)).fillna(50).values

    # Feature 3: MACD histogram (relative)
    ema12 = close_series.ewm(span=12, adjust=False).mean()
    ema26 = close_series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    features['macd'] = (macd_line - macd_line.ewm(span=9, adjust=False).mean()).fillna(0).values

    # Feature 4: volatility (rolling std of returns)
    features['volatility'] = pd.Series(features['returns'].values).rolling(
        14, min_periods=1).std().fillna(0).values

    # Feature 5: Bollinger %B (normalized position 0-1)
    sma20 = close_series.rolling(20, min_periods=1).mean()
    std20 = close_series.rolling(20, min_periods=1).std().fillna(0)
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    features['bb_pctb'] = ((close_series - bb_lower) / (bb_upper - bb_lower + 1e-10)).fillna(0.5).clip(-0.5, 1.5).values

    # Feature 6: Bollinger bandwidth (relative volatility measure)
    features['bb_bw'] = ((bb_upper - bb_lower) / (sma20 + 1e-10)).fillna(0).values

    # Feature 7: candle body ratio (normalized)
    candle_range = high_arr - low_arr + 1e-10
    features['body_ratio'] = ((close_arr - open_arr) / candle_range).clip(-1, 1)

    # Feature 8: ATR normalized by price (relative volatility)
    high_s = pd.Series(high_arr)
    low_s = pd.Series(low_arr)
    tr = pd.concat([
        high_s - low_s,
        (high_s - close_series.shift(1)).abs(),
        (low_s - close_series.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14, min_periods=1).mean()
    features['atr_norm'] = (atr_14 / (close_series + 1e-10)).fillna(0).values

    # Feature 9: momentum (rate of change over 10 periods)
    features['momentum'] = close_series.pct_change(10).fillna(0).clip(-0.2, 0.2).values

    # Feature 10: EMA crossover signal (normalized)
    features['ema_cross'] = ((ema12 - ema26) / (close_series + 1e-10)).fillna(0).values

    # Feature 11: regime label (ATR quartiles x trend)
    atr_q = pd.qcut(atr_14.rank(method='first'), q=4, labels=False).fillna(0)
    sma50 = close_series.rolling(50, min_periods=1).mean()
    trend_up = (close_series > sma50).astype(int)
    regime = np.zeros(len(close_arr), dtype=np.float32)
    regime[(atr_q <= 1) & (trend_up == 0)] = 0
    regime[(atr_q <= 1) & (trend_up == 1)] = 1
    regime[(atr_q >= 2) & (trend_up == 1)] = 2
    regime[(atr_q >= 2) & (trend_up == 0)] = 3
    features['regime'] = regime

    print(f"[*] Total features: {features.shape[1]} (relative only, no absolute prices)")
    print(f"    log_vol, returns, rsi, macd, volatility, bb_pctb, bb_bw,")
    print(f"    body_ratio, atr_norm, momentum, ema_cross, regime")

    close_prices = close_arr
    feature_matrix = features.values

    # Create windows with multi-step horizon labels
    print(f"[*] Windows: size={window_size}, horizon={horizon}, threshold={threshold}")
    X, y_dir, weights = [], [], []

    for i in range(window_size, len(feature_matrix) - horizon):
        X.append(feature_matrix[i - window_size: i])
        cur = close_prices[i]
        fut = close_prices[i + horizon]
        pct = (fut - cur) / cur
        y_dir.append(1 if fut > cur else 0)
        weights.append(1.0 if abs(pct) > threshold else 0.5)

    X = np.array(X, dtype=np.float32)
    y_dir = np.array(y_dir, dtype=np.int64)
    weights = np.array(weights, dtype=np.float32)

    print(f"[OK] Samples: {len(y_dir)}, Up: {np.mean(y_dir):.2%}, Down: {1-np.mean(y_dir):.2%}")

    # 80/20 split
    split = int(len(y_dir) * split_ratio)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y_dir[:split], y_dir[split:]
    w_tr, w_te = weights[:split], weights[split:]
    test_close = close_prices[split + window_size: split + window_size + len(y_te)]

    # Scale features
    scaler = StandardScaler()
    n_tr, w, f = X_tr.shape
    X_tr = scaler.fit_transform(X_tr.reshape(-1, f)).reshape(n_tr, w, f).astype(np.float32)
    X_te = scaler.transform(X_te.reshape(-1, f)).reshape(len(y_te), w, f).astype(np.float32)

    print(f"[OK] Train: {len(y_tr)}, Test: {len(y_te)}, Purge gap: {purge_gap}")
    return X_tr, X_te, y_tr, y_te, w_tr, w_te, test_close, scaler, purge_gap


# ============================================================================
# Training & Validation
# ============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, accum=4):
    model.train()
    total_loss, correct, total, n = 0, 0, 0, 0
    optimizer.zero_grad(set_to_none=True)

    for step, (xb, yb, wb) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).float().unsqueeze(1)

        # Add small noise augmentation during training for regularization
        if model.training:
            xb = xb + torch.randn_like(xb) * 0.01

        with autocast(device_type='cuda', dtype=torch.float16):
            logits = model(xb)
        # Compute loss in float32 to prevent NaN from AMP
        logits_f32 = logits.float()
        loss = criterion(logits_f32, yb) / accum

        scaler.scale(loss).backward()

        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accum
        n += 1
        preds = (torch.sigmoid(logits) > 0.5).int()
        correct += (preds == yb.int()).sum().item()
        total += yb.size(0)

    return total_loss / n, correct / total * 100


@torch.no_grad()
def validate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss, n = 0, 0
    all_probs, all_tgt = [], []

    for xb, yb, wb in loader:
        xb = xb.to(device)
        yb_t = yb.to(device).float().unsqueeze(1)
        logits = model(xb)
        total_loss += criterion(logits, yb_t).item()
        n += 1
        all_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
        all_tgt.extend(yb.numpy().flatten())

    probs = np.array(all_probs)
    tgt = np.array(all_tgt)
    preds = (probs > threshold).astype(int)

    return (total_loss / n,
            accuracy_score(tgt, preds) * 100,
            precision_score(tgt, preds, zero_division=0) * 100,
            recall_score(tgt, preds, zero_division=0) * 100,
            f1_score(tgt, preds, zero_division=0) * 100,
            probs, tgt)


def optimize_threshold(probs, targets, target_recall=0.80):
    """Find threshold that maximizes precision while keeping recall >= target."""
    best_t, best_prec = 0.5, 0.0
    for t in np.arange(0.20, 0.90, 0.005):
        p = (probs > t).astype(int)
        rec = recall_score(targets, p, zero_division=0)
        prec = precision_score(targets, p, zero_division=0)

        if rec >= target_recall:
            if prec > best_prec:
                best_prec, best_t = prec, t

    # Fallback to maximizing F1 if no threshold gives target recall
    if best_prec == 0.0:
        best_s = 0
        for t in np.arange(0.20, 0.90, 0.005):
            p = (probs > t).astype(int)
            s = f1_score(targets, p, zero_division=0)
            if s > best_s:
                best_s, best_t = s, t
        return best_t, best_s

    return best_t, best_prec


# ============================================================================
# Plotting
# ============================================================================

def plot_training_history(history, save_path='wmc_training_history.png'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ep = range(1, len(history['train_loss']) + 1)

    axes[0, 0].plot(ep, history['train_loss'], label='Train', lw=2)
    axes[0, 0].plot(ep, history['val_loss'], label='Val', lw=2)
    axes[0, 0].set_title('Classification Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(ep, history['train_acc'], label='Train', lw=2, color='green')
    axes[0, 1].plot(ep, history['val_acc'], label='Val', lw=2, color='orange')
    axes[0, 1].axhline(y=50, color='red', ls='--', alpha=0.5, label='Random')
    axes[0, 1].set_title('Accuracy (%)', fontsize=14, fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(ep, history['val_precision'], label='Precision', lw=2)
    axes[1, 0].plot(ep, history['val_recall'], label='Recall', lw=2)
    axes[1, 0].plot(ep, history['val_f1'], label='F1', lw=2)
    axes[1, 0].axhline(y=80, color='red', ls='--', alpha=0.5, label='Target 80%')
    axes[1, 0].set_title('Metrics (%)', fontsize=14, fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(ep, history['lr'], lw=2, color='purple')
    axes[1, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
    axes[1, 1].set_yscale('log'); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[OK] {save_path}")


def plot_predictions(test_close, preds, targets, save_path='wmc_predictions.png'):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    n = min(len(test_close), len(preds))
    x = np.arange(n)
    correct = preds[:n] == targets[:n]

    axes[0].plot(x, test_close[:n], 'b-', lw=1, alpha=0.7, label='BTC Price')
    buy_ok = (preds[:n] == 1) & correct
    buy_bad = (preds[:n] == 1) & ~correct
    sell_ok = (preds[:n] == 0) & correct
    axes[0].scatter(x[buy_ok], test_close[:n][buy_ok], marker='^', c='green',
                     s=8, alpha=0.4, label='Buy (Correct)')
    axes[0].scatter(x[buy_bad], test_close[:n][buy_bad], marker='^', c='red',
                     s=8, alpha=0.3, label='Buy (Wrong)')
    axes[0].set_title('Bitcoin Price with Buy/Sell Signals (Wavelet-Mamba Classifier v2)',
                       fontsize=15, fontweight='bold')
    axes[0].set_xlabel('Time Steps'); axes[0].set_ylabel('Price (USD)')
    axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.3)

    w = max(10, min(100, n // 20))
    roll = np.convolve(correct.astype(float), np.ones(w) / w, mode='valid')
    axes[1].plot(roll * 100, lw=2, color='blue')
    axes[1].axhline(y=50, color='red', ls='--', alpha=0.5, label='Random 50%')
    axes[1].axhline(y=80, color='green', ls='--', alpha=0.5, label='Target 80%')
    axes[1].fill_between(range(len(roll)), 50, roll * 100,
                          where=(roll * 100 > 50), color='green', alpha=0.2)
    axes[1].set_title(f'Rolling Accuracy (w={w})', fontsize=15, fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[OK] {save_path}")


def plot_directional_analysis(targets, preds, probs, save_path='wmc_directional_analysis.png'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    correct = preds == targets
    x = np.arange(len(targets))

    axes[0, 0].scatter(x[correct], targets[correct], c='green', marker='o', s=5, alpha=0.3, label='Correct')
    axes[0, 0].scatter(x[~correct], targets[~correct], c='red', marker='x', s=5, alpha=0.3, label='Incorrect')
    axes[0, 0].set_title('Prediction Correctness', fontsize=14, fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(probs[targets == 1], bins=50, alpha=0.6, color='green', label='Up', density=True)
    axes[0, 1].hist(probs[targets == 0], bins=50, alpha=0.6, color='red', label='Down', density=True)
    axes[0, 1].axvline(x=0.5, color='black', ls='--', label='Threshold')
    axes[0, 1].set_title('Probability Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)
    axes[1, 0].plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC={roc_auc:.3f})')
    axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[1, 0].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('FPR'); axes[1, 0].set_ylabel('TPR')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    w = max(10, min(100, len(correct) // 20))
    roll = np.convolve(correct.astype(float), np.ones(w) / w, mode='valid')
    axes[1, 1].plot(roll * 100, lw=2, color='blue')
    axes[1, 1].axhline(y=50, color='red', ls='--', label='Random')
    axes[1, 1].axhline(y=80, color='green', ls='--', label='Target 80%')
    axes[1, 1].fill_between(range(len(roll)), 50, roll * 100,
                              where=(roll * 100 > 50), color='green', alpha=0.2)
    axes[1, 1].set_title(f'Rolling Accuracy (w={w})', fontsize=14, fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[OK] {save_path}")


def plot_classification_metrics(metrics, cm, save_path='wmc_classification_metrics.png'):
    fig = plt.figure(figsize=(15, 6))
    ax1 = plt.subplot(1, 2, 1)
    names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    vals = [metrics['accuracy'] / 100, metrics['precision'] / 100,
            metrics['recall'] / 100, metrics['f1'] / 100]
    colors = ['#4ECDC4', '#45B7D1', '#FF6B6B', '#FFA07A']
    bars = ax1.bar(names, vals, color=colors, edgecolor='white', linewidth=1.5)
    ax1.axhline(y=0.8, color='red', ls='--', alpha=0.5, label='Target 80%')
    ax1.set_ylim(0, 1.15)
    ax1.set_title('Binary Classification Metrics', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3); ax1.legend()
    for b in bars:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2., h + 0.02, f'{h:.3f}',
                 ha='center', fontweight='bold')

    ax2 = plt.subplot(1, 2, 2)
    im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Sell (0)', 'Buy (1)'])
    ax2.set_yticklabels(['Sell (0)', 'Buy (1)'])
    thresh_cm = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                     color="white" if cm[i, j] > thresh_cm else "black",
                     fontweight='bold', fontsize=12)
    ax2.set_ylabel('Actual'); ax2.set_xlabel('Predicted')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[OK] {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("BITCOIN BUY/SELL — WAVELET-MAMBA BINARY CLASSIFIER v2")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")

    # ===== Fine-tuned Hyperparameters v2 =====
    WINDOW_SIZE = 64
    HORIZON = 4           # predict 1-hour ahead direction
    THRESHOLD = 0.001     # 0.1% min movement for clear signal
    EPOCHS = 120
    BATCH_SIZE = 256
    LR = 1.5e-4           # lower for stability
    WEIGHT_DECAY = 5e-4   # stronger regularization
    ACCUM_STEPS = 4
    PATIENCE = 40
    VAL_SPLIT = 0.1
    D_MODEL = 128
    D_STATE = 16
    DROPOUT = 0.25        # higher to reduce overfitting
    SWA_FRAC = 0.75
    GAMMA_POS = 1.0
    GAMMA_NEG = 4.0
    PURGE_GAP = 256
    FBETA = 0.5

    # ===== Preprocess =====
    (X_train, X_test, y_train, y_test,
     w_train, w_test, test_close, scaler, purge_gap) = preprocess_data(
        window_size=WINDOW_SIZE, horizon=HORIZON, threshold=THRESHOLD,
        purge_gap=PURGE_GAP)

    n_features = X_train.shape[2]
    n_pos = np.sum(y_train)
    n_neg = len(y_train) - n_pos
    alpha = n_neg / (n_pos + n_neg)
    print(f"[*] Class balance - Pos: {n_pos}, Neg: {n_neg}, Alpha: {alpha:.3f}")
    print(f"[*] Features: {n_features}")

    # ===== Data Loaders (with purged gap) =====
    n_tr = int(len(y_train) * (1 - VAL_SPLIT)) - PURGE_GAP

    train_ds = TensorDataset(torch.tensor(X_train[:n_tr], dtype=torch.float32),
                              torch.tensor(y_train[:n_tr], dtype=torch.long),
                              torch.tensor(w_train[:n_tr], dtype=torch.float32))
    # Skip purge_gap samples between train and val
    val_start = n_tr + PURGE_GAP
    val_ds = TensorDataset(torch.tensor(X_train[val_start:], dtype=torch.float32),
                            torch.tensor(y_train[val_start:], dtype=torch.long),
                            torch.tensor(w_train[val_start:], dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.long),
                             torch.tensor(w_test, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=True, num_workers=0)

    print(f"[*] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, "
          f"Test samples: {len(test_ds)}")

    # ===== Build Model =====
    model = WaveletMambaClassifier(
        n_features=n_features, window_size=WINDOW_SIZE,
        d_model=D_MODEL, d_state=D_STATE, dropout=DROPOUT
    ).to(device)
    print(f"[*] Parameters: {model.count_parameters():,}")

    # ===== Loss =====
    criterion = CombinedClassificationLoss(
        alpha=alpha, gamma_pos=GAMMA_POS, gamma_neg=GAMMA_NEG,
        label_smoothing=0.1, beta=FBETA,
        w_focal=0.6, w_fbeta=0.4
    )

    # ===== Optimizer with Layer-wise LR =====
    # Classification head gets 3× the learning rate of the backbone
    backbone_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if any(k in name for k in ['classifier', 'output', 'gate', 'regime_embed', 'cls_norm']):
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': LR},
        {'params': classifier_params, 'lr': LR * 3},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-7)

    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5, anneal_epochs=5)
    swa_start = int(EPOCHS * SWA_FRAC)

    amp_scaler = GradScaler('cuda')

    print(f"\n{'='*80}")
    print(f"  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR} (cls head: {LR*3})")
    print(f"  Loss: Combined(Asym[a={alpha:.3f},g+={GAMMA_POS},g-={GAMMA_NEG}] + Fbeta[b={FBETA}])")
    print(f"  Label smoothing: 0.1, Accum steps: {ACCUM_STEPS}")
    print(f"  Scheduler: CosineWarmRestarts + SWA from epoch {swa_start}")
    print(f"  Horizon: {HORIZON} candles, Threshold: {THRESHOLD}")
    print(f"  Purge gap: {PURGE_GAP}, Patience: {PATIENCE}")
    print(f"{'='*80}\n")

    # ===== Training Loop =====
    history = {k: [] for k in ['train_loss', 'val_loss', 'train_acc', 'val_acc',
                                 'val_precision', 'val_recall', 'val_f1', 'lr']}
    best_f1, best_epoch, patience_counter = 0, 0, 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, amp_scaler, ACCUM_STEPS)

        val_loss, val_acc, val_prec, val_rec, val_f1, val_probs, val_tgt = \
            validate(model, val_loader, criterion, device)

        # Scheduler
        if epoch < swa_start:
            scheduler.step()
        else:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        history['val_f1'].append(val_f1)
        history['lr'].append(lr)

        print(f"  Epoch {epoch:>3}/{EPOCHS} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"Acc: {train_acc:.1f}/{val_acc:.1f}% | "
              f"P/R/F1: {val_prec:.1f}/{val_rec:.1f}/{val_f1:.1f}% | "
              f"LR: {lr:.2e} | {dt:.1f}s")

        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            torch.save(model.state_dict(), 'wmc_best_model.pth')
            patience_counter = 0
            print(f"      [BEST] New best F1: {best_f1:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[!] Early stopping at epoch {epoch}")
                break

    # ===== Update BN for SWA =====
    print("\n[*] Updating SWA batch normalization...")
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    torch.save(swa_model.module.state_dict(), 'wmc_swa_model.pth')

    # ===== Load Best Model =====
    model.load_state_dict(torch.load('wmc_best_model.pth', weights_only=True))

    # Compare regular vs SWA on validation
    _, _, _, _, f1_reg, _, _ = validate(model, val_loader, criterion, device)
    swa_model_eval = WaveletMambaClassifier(
        n_features=n_features, window_size=WINDOW_SIZE,
        d_model=D_MODEL, d_state=D_STATE, dropout=DROPOUT).to(device)
    swa_model_eval.load_state_dict(torch.load('wmc_swa_model.pth', weights_only=True))
    _, _, _, _, f1_swa, _, _ = validate(swa_model_eval, val_loader, criterion, device)

    print(f"\n  Regular best F1: {f1_reg:.2f}% (epoch {best_epoch})")
    print(f"  SWA F1:         {f1_swa:.2f}%")

    if f1_swa > f1_reg:
        print("  -> Using SWA model")
        final_model = swa_model_eval
    else:
        print("  -> Using regular best model")
        final_model = model

    torch.save(final_model.state_dict(), 'wmc_final_model.pth')

    # ===== Threshold Optimization on Validation =====
    print("\n[*] Optimizing classification threshold...")
    _, _, _, _, _, val_probs, val_tgt = validate(final_model, val_loader, criterion, device)
    opt_thresh, opt_score = optimize_threshold(val_probs, val_tgt)
    print(f"  Optimal threshold: {opt_thresh:.3f} (score: {opt_score:.4f})")

    # ===== Test Evaluation =====
    print("\n" + "=" * 80)
    print("EVALUATION ON TEST SET")
    print("=" * 80)

    test_loss, _, _, _, _, test_probs, test_tgt = \
        validate(final_model, test_loader, criterion, device, threshold=opt_thresh)
    test_preds = (test_probs > opt_thresh).astype(int)

    acc = accuracy_score(test_tgt, test_preds) * 100
    prec = precision_score(test_tgt, test_preds, zero_division=0) * 100
    rec = recall_score(test_tgt, test_preds, zero_division=0) * 100
    f1 = f1_score(test_tgt, test_preds, zero_division=0) * 100
    cm = confusion_matrix(test_tgt, test_preds)

    print(f"\n  Threshold: {opt_thresh:.3f}")
    print(f"  Accuracy:  {acc:.2f}%")
    print(f"  Precision: {prec:.2f}%")
    print(f"  Recall:    {rec:.2f}%")
    print(f"  F1 Score:  {f1:.2f}%")
    print(f"\n  Confusion Matrix:")
    print(f"               Predicted Sell   Predicted Buy")
    print(f"  Actual Sell   {cm[0][0]:<14} {cm[0][1]}")
    print(f"  Actual Buy    {cm[1][0]:<14} {cm[1][1]}")

    print(f"\n{classification_report(test_tgt, test_preds, target_names=['Sell', 'Buy'])}")

    # ===== Plots =====
    print("\n[*] Generating plots...")
    plot_training_history(history)
    metrics = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    plot_predictions(test_close, test_preds, test_tgt.astype(int))
    plot_directional_analysis(test_tgt.astype(int), test_preds, test_probs)
    plot_classification_metrics(metrics, cm)

    # ===== Save Results =====
    results = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1,
        'Optimal_Threshold': opt_thresh, 'Horizon': HORIZON,
        'Confusion_Matrix': cm, 'Best_Epoch': best_epoch,
    }
    np.save('wmc_evaluation_results.npy', results)

    print("\n" + "=" * 80)
    print("[OK] BINARY CLASSIFICATION v2 COMPLETE!")
    print("=" * 80)
    print("\n  Generated Files:")
    print("    wmc_best_model.pth             - Best model")
    print("    wmc_swa_model.pth              - SWA model")
    print("    wmc_final_model.pth            - Final model")
    print("    wmc_training_history.png       - Training curves")
    print("    wmc_predictions.png            - Price + signals")
    print("    wmc_directional_analysis.png   - Directional analysis + ROC")
    print("    wmc_classification_metrics.png - Metrics + confusion matrix")
    print("    wmc_evaluation_results.npy     - All metrics")

    status = "[PASS]" if all(v >= 80 for v in [acc, prec, rec, f1]) else "[BELOW TARGET]"
    print(f"\n  {status}: Acc={acc:.1f}% P={prec:.1f}% R={rec:.1f}% F1={f1:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
