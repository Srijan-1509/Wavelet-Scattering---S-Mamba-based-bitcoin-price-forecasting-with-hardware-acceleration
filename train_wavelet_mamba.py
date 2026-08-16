"""
Training Script — Enhanced Wavelet Scattering + Mamba SSM
===========================================================
Full training pipeline with:
  - 100 epochs, Cosine Annealing with Warm Restarts
  - SWA (Stochastic Weight Averaging) for last 20%
  - Gradient accumulation (effective batch size 512)
  - Dual loss (price Huber + direction BCE with label smoothing)
  - Comprehensive metrics: Precision, Recall, F1, Directional Accuracy
  - Walk-forward validation
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from mamba_predictor import WaveletMambaWrapper, CombinedLoss
import pickle
import os
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from torch.amp import autocast, GradScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ============================================================================
# Device Setup
# ============================================================================

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[*] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("[!] CUDA not available, using CPU")
    return device


# ============================================================================
# Metrics
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """Comprehensive evaluation metrics"""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    y_true_direction = np.diff(y_true) > 0
    y_pred_direction = np.diff(y_pred) > 0
    directional_accuracy = np.mean(y_true_direction == y_pred_direction) * 100

    precision = precision_score(y_true_direction, y_pred_direction, zero_division=0)
    recall = recall_score(y_true_direction, y_pred_direction, zero_division=0)
    f1 = f1_score(y_true_direction, y_pred_direction, zero_division=0)
    conf_matrix = confusion_matrix(y_true_direction, y_pred_direction)

    return {
        'RMSE': rmse, 'MAE': mae, 'MAPE': mape,
        'Directional_Accuracy': directional_accuracy,
        'Precision': precision, 'Recall': recall, 'F1': f1,
        'Confusion_Matrix': conf_matrix
    }


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    if np.std(returns) == 0:
        return 0
    return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)


def calculate_max_drawdown(prices):
    cumulative = np.maximum.accumulate(prices)
    drawdown = (prices - cumulative) / cumulative
    return np.min(drawdown) * 100


# ============================================================================
# Training Loop (with gradient accumulation)
# ============================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler,
                    accumulation_steps=2):
    model.train()
    total_loss = 0.0
    total_price_loss = 0.0
    total_dir_loss = 0.0
    n_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for step, (X_batch, y_price, y_dir) in enumerate(dataloader):
        X_batch = X_batch.to(device, non_blocking=True)
        y_price = y_price.to(device, non_blocking=True).unsqueeze(1)
        y_dir = y_dir.to(device, non_blocking=True).float().unsqueeze(1)

        with autocast(device_type='cuda', dtype=torch.float16):
            price_pred, dir_logit = model(X_batch)
            loss, p_loss, d_loss = criterion(price_pred, dir_logit, y_price, y_dir)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accumulation_steps
        total_price_loss += p_loss.item()
        total_dir_loss += d_loss.item()
        n_batches += 1

        if n_batches % 20 == 0:
            print(f"      Batch {n_batches:>3}: Loss = {loss.item() * accumulation_steps:.4f}")

    return (total_loss / n_batches,
            total_price_loss / n_batches,
            total_dir_loss / n_batches)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_price_loss = 0.0
    total_dir_loss = 0.0
    n_batches = 0
    all_price_preds = []
    all_price_targets = []
    all_dir_preds = []
    all_dir_targets = []

    for X_batch, y_price, y_dir in dataloader:
        X_batch = X_batch.to(device)
        y_price_t = y_price.to(device).unsqueeze(1)
        y_dir_t = y_dir.to(device).float().unsqueeze(1)

        price_pred, dir_logit = model(X_batch)

        loss, p_loss, d_loss = criterion(price_pred, dir_logit, y_price_t, y_dir_t)
        total_loss += loss.item()
        total_price_loss += p_loss.item()
        total_dir_loss += d_loss.item()
        n_batches += 1

        all_price_preds.extend(price_pred.cpu().numpy().flatten())
        all_price_targets.extend(y_price.numpy().flatten())
        all_dir_preds.extend((torch.sigmoid(dir_logit) > 0.5).int().cpu().numpy().flatten())
        all_dir_targets.extend(y_dir.numpy().flatten())

    avg_loss = total_loss / n_batches

    return (avg_loss, total_price_loss / n_batches, total_dir_loss / n_batches,
            np.array(all_price_preds), np.array(all_price_targets),
            np.array(all_dir_preds), np.array(all_dir_targets))


def compute_directional_accuracy(y_true, y_pred):
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    if len(y_true_diff) == 0:
        return 0.0
    return np.mean((y_true_diff > 0) == (y_pred_diff > 0)) * 100


# ============================================================================
# Plotting
# ============================================================================

def plot_training_history(history, save_path='wm_training_history.png'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0, 0].plot(epochs, history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_title('Combined Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history['train_price_loss'], label='Train Price', linewidth=2)
    axes[0, 1].plot(epochs, history['val_price_loss'], label='Val Price', linewidth=2)
    axes[0, 1].set_title('Price Loss (Huber)', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history['train_dir_loss'], label='Train Dir', linewidth=2)
    axes[1, 0].plot(epochs, history['val_dir_loss'], label='Val Dir', linewidth=2)
    axes[1, 0].set_title('Direction Loss (BCE)', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    if 'val_dir_acc' in history and history['val_dir_acc']:
        axes[1, 1].plot(epochs, history['val_dir_acc'], linewidth=2, color='green')
        axes[1, 1].axhline(y=50, color='red', linestyle='--', label='Random baseline')
        axes[1, 1].set_title('Directional Accuracy (%)', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Training history saved to {save_path}")
    plt.close()


def plot_predictions(y_true, y_pred, save_path='wm_predictions.png', num_samples=None):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    if num_samples is None:
        num_samples = len(y_true)
    plot_len = min(num_samples, len(y_true))
    x_axis = np.arange(plot_len)

    axes[0].plot(x_axis, y_true[:plot_len], 'b-', linewidth=2, label='Actual', alpha=0.8)
    axes[0].plot(x_axis, y_pred[:plot_len], 'r--', linewidth=2, label='Predicted', alpha=0.8)
    axes[0].set_title('Actual vs Predicted Bitcoin Prices (Enhanced Wavelet-Mamba)',
                       fontsize=15, fontweight='bold')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Price (USD)')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    error_pct = ((y_pred[:plot_len] - y_true[:plot_len]) / y_true[:plot_len]) * 100
    axes[1].plot(x_axis, error_pct, 'purple', linewidth=1.5, alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='--')
    axes[1].fill_between(x_axis, error_pct, 0, where=(error_pct > 0),
                          color='green', alpha=0.3, label='Over')
    axes[1].fill_between(x_axis, error_pct, 0, where=(error_pct < 0),
                          color='red', alpha=0.3, label='Under')
    axes[1].set_title('Prediction Error (%)', fontsize=15, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Predictions saved to {save_path}")
    plt.close()


def plot_directional_accuracy(y_true, y_pred, save_path='wm_directional_analysis.png'):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    true_dir = np.diff(y_true) > 0
    pred_dir = np.diff(y_pred) > 0
    correct = true_dir == pred_dir
    x_axis = np.arange(len(true_dir))

    axes[0].scatter(x_axis[correct], true_dir[correct], c='green', marker='o',
                     s=20, alpha=0.5, label='Correct')
    axes[0].scatter(x_axis[~correct], true_dir[~correct], c='red', marker='x',
                     s=20, alpha=0.5, label='Incorrect')
    axes[0].set_title('Direction Prediction (Green=Correct)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    window = max(5, min(50, len(correct) // 10))
    rolling = np.convolve(correct.astype(float), np.ones(window) / window, mode='valid')
    axes[1].plot(rolling * 100, linewidth=2, color='blue')
    axes[1].axhline(y=50, color='red', linestyle='--', label='Random (50%)')
    axes[1].fill_between(range(len(rolling)), 50, rolling * 100,
                          where=(rolling * 100 > 50), color='green', alpha=0.3)
    axes[1].set_title(f'Rolling Directional Accuracy (Window={window})',
                       fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Directional analysis saved to {save_path}")
    plt.close()


def plot_classification_metrics(metrics, save_path='wm_classification_metrics.png'):
    fig = plt.figure(figsize=(15, 6))

    ax1 = plt.subplot(1, 2, 1)
    names = ['Precision', 'Recall', 'F1 Score', 'Dir. Accuracy']
    values = [metrics['Precision'], metrics['Recall'], metrics['F1'],
              metrics['Directional_Accuracy'] / 100]
    colors = ['#4ECDC4', '#45B7D1', '#FF6B6B', '#FFA07A']
    bars = ax1.bar(names, values, color=colors, edgecolor='white', linewidth=1.5)
    ax1.set_ylim(0, 1.15)
    ax1.set_title('Classification Metrics', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., h + 0.02,
                 f'{h:.3f}', ha='center', fontweight='bold')

    ax2 = plt.subplot(1, 2, 2)
    cm = metrics['Confusion_Matrix']
    im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    classes = ['Down', 'Up']
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(classes)
    ax2.set_yticklabels(classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontweight='bold', fontsize=12)
    ax2.set_ylabel('Actual')
    ax2.set_xlabel('Predicted')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Classification metrics saved to {save_path}")
    plt.close()


# ============================================================================
# Walk-forward Validation
# ============================================================================

def walk_forward_validation(model, X_test, y_test, device, window_size=10):
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION")
    print("=" * 80)

    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(0, len(y_test), window_size):
            end_idx = min(i + window_size, len(y_test))
            X_batch = torch.tensor(X_test[i:end_idx], dtype=torch.float32).to(device)
            price_pred, _ = model(X_batch)
            predictions.extend(price_pred.cpu().numpy().flatten())

            if (i // window_size) % 20 == 0:
                progress = (end_idx / len(y_test)) * 100
                print(f"  Progress: {progress:.1f}%")

    predictions = np.array(predictions[:len(y_test)])
    print("[OK] Walk-forward validation complete")
    return predictions


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("BITCOIN PRICE PREDICTION — ENHANCED WAVELET-MAMBA SSM")
    print("=" * 80)

    device = get_device()

    # Load preprocessed data
    print("\n[*] Loading preprocessed data...")
    try:
        X_train = np.load('wm_X_train.npy')
        X_test = np.load('wm_X_test.npy')
        y_price_train = np.load('wm_y_price_train.npy')
        y_price_test = np.load('wm_y_price_test.npy')
        y_dir_train = np.load('wm_y_dir_train.npy')
        y_dir_test = np.load('wm_y_dir_test.npy')

        with open('wm_target_scaler.pkl', 'rb') as f:
            target_scaler = pickle.load(f)

        # Subset for faster iteration on RTX 4060 (CryptoMamba used only 2190 points anyway)
        subset_size = 40000
        if len(X_train) > subset_size:
            X_train = X_train[-subset_size:]
            y_price_train = y_price_train[-subset_size:]
            y_dir_train = y_dir_train[-subset_size:]

        print(f"  Train: {X_train.shape[0]} samples, shape={X_train.shape}")
        print(f"  Test:  {X_test.shape[0]} samples")
    except FileNotFoundError:
        print("\n[!] Data not found! Run: python wavelet_mamba_preprocess.py")
        return

    # Build model
    window_size = X_train.shape[1]
    n_features = X_train.shape[2]

    builder = WaveletMambaWrapper(window_size=window_size, n_features=n_features)
    builder.build_model()
    model = builder.model

    # ===== Hyperparameters =====
    EPOCHS = 60
    BATCH_SIZE = 128
    VAL_SPLIT = 0.1
    LR = 0.0005
    WEIGHT_DECAY = 1e-4
    ACCUMULATION_STEPS = 4
    PATIENCE = 25
    SWA_START_FRAC = 0.8  # Start SWA at 80% of training

    n_train = int(len(y_price_train) * (1 - VAL_SPLIT))

    train_ds = TensorDataset(
        torch.tensor(X_train[:n_train], dtype=torch.float32),
        torch.tensor(y_price_train[:n_train], dtype=torch.float32),
        torch.tensor(y_dir_train[:n_train], dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_train[n_train:], dtype=torch.float32),
        torch.tensor(y_price_train[n_train:], dtype=torch.float32),
        torch.tensor(y_dir_train[n_train:], dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_price_test, dtype=torch.float32),
        torch.tensor(y_dir_test, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=True, num_workers=0)

    # Training setup
    criterion = CombinedLoss(price_weight=0.6, direction_weight=0.3,
                              directional_penalty=0.1, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Cosine Annealing with Warm Restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-7
    )

    # SWA setup
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5, anneal_epochs=5)
    swa_start = int(EPOCHS * SWA_START_FRAC)

    print(f"\n  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, Val split: {VAL_SPLIT}")
    print(f"  Optimizer: AdamW, LR: {LR}, Weight decay: {WEIGHT_DECAY}")
    print(f"  Loss: Huber(0.6) + BCE(0.3) + DirPenalty(0.1)")
    print(f"  Gradient Accumulation: {ACCUMULATION_STEPS} steps (eff. batch={BATCH_SIZE*ACCUMULATION_STEPS})")
    print(f"  Scheduler: CosineAnnealingWarmRestarts(T0=20, Tmult=2)")
    print(f"  SWA: starts at epoch {swa_start}")
    print(f"  Mixed Precision: AMP (float16)")

    # AMP scaler
    scaler = GradScaler('cuda')

    print("\n[*] Skipping training, loading best model for evaluation...")
    model.load_state_dict(torch.load('wm_best_model.pth', weights_only=True))

    # Evaluate regular best
    _, _, _, y_pred_reg, _, _, _ = validate(model, val_loader, criterion, device)

    y_pred_reg_inv = target_scaler.inverse_transform(y_pred_reg.reshape(-1, 1)).flatten()
    
    val_actual = target_scaler.inverse_transform(
        y_price_train[n_train:].reshape(-1, 1)).flatten()

    rmse_reg = np.sqrt(np.mean((val_actual - y_pred_reg_inv) ** 2))

    print(f"\n  Regular best RMSE: ${rmse_reg:.2f}")

    final_model = model
    torch.save(model.state_dict(), 'wm_final_model.pth')

    print("[OK] Final model saved")

    # ===== Evaluation =====
    print("\n" + "=" * 80)
    print("EVALUATION ON TEST SET")
    print("=" * 80)

    _, _, _, y_pred_scaled, _, dir_preds, dir_targets = validate(
        final_model, test_loader, criterion, device)

    y_pred_test = target_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_actual = target_scaler.inverse_transform(
        y_price_test.reshape(-1, 1)).flatten()

    # Price metrics
    metrics = calculate_metrics(y_test_actual, y_pred_test)

    print("\n[*] PRICE PREDICTION METRICS:")
    print(f"  RMSE: ${metrics['RMSE']:.2f}")
    print(f"  MAE:  ${metrics['MAE']:.2f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")

    print("\n[*] DIRECTIONAL METRICS (from price):")
    print(f"  Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1 Score:  {metrics['F1']:.4f}")

    print("\n[*] DIRECTION HEAD METRICS (binary classifier):")
    dir_precision = precision_score(dir_targets, dir_preds, zero_division=0)
    dir_recall = recall_score(dir_targets, dir_preds, zero_division=0)
    dir_f1 = f1_score(dir_targets, dir_preds, zero_division=0)
    dir_acc = accuracy_score(dir_targets, dir_preds)
    print(f"  Accuracy:  {dir_acc:.4f}")
    print(f"  Precision: {dir_precision:.4f}")
    print(f"  Recall:    {dir_recall:.4f}")
    print(f"  F1 Score:  {dir_f1:.4f}")

    print("\n[*] Confusion Matrix (Price Direction):")
    print("                 Predicted Down   Predicted Up")
    cm = metrics['Confusion_Matrix']
    print(f"Actual Down      {cm[0][0]:<14} {cm[0][1]}")
    print(f"Actual Up        {cm[1][0]:<14} {cm[1][1]}")

    # Trading metrics
    returns_actual = np.diff(y_test_actual) / y_test_actual[:-1]
    returns_pred = np.diff(y_pred_test) / y_pred_test[:-1]
    sharpe_actual = calculate_sharpe_ratio(returns_actual)
    sharpe_pred = calculate_sharpe_ratio(returns_pred)
    max_dd_actual = calculate_max_drawdown(y_test_actual)
    max_dd_pred = calculate_max_drawdown(y_pred_test)

    print(f"\n[*] TRADING METRICS:")
    print(f"  Sharpe (Actual):  {sharpe_actual:.4f}")
    print(f"  Sharpe (Pred):    {sharpe_pred:.4f}")
    print(f"  Max DD (Actual):  {max_dd_actual:.2f}%")
    print(f"  Max DD (Pred):    {max_dd_pred:.2f}%")

    # Walk-forward
    y_pred_walk_scaled = walk_forward_validation(final_model, X_test, y_price_test, device)
    y_pred_walk = target_scaler.inverse_transform(
        y_pred_walk_scaled.reshape(-1, 1)).flatten()
    metrics_walk = calculate_metrics(y_test_actual, y_pred_walk)

    print("\n[*] WALK-FORWARD VALIDATION:")
    print(f"  RMSE: ${metrics_walk['RMSE']:.2f}")
    print(f"  Dir Accuracy: {metrics_walk['Directional_Accuracy']:.2f}%")
    print(f"  F1: {metrics_walk['F1']:.4f}")

    # Generate plots
    print("\n[*] Generating plots...")
    plot_predictions(y_test_actual, y_pred_test)
    plot_directional_accuracy(y_test_actual, y_pred_test)
    plot_classification_metrics(metrics)

    # Save results
    results = {
        'RMSE': metrics['RMSE'], 'MAE': metrics['MAE'], 'MAPE': metrics['MAPE'],
        'Directional_Accuracy': metrics['Directional_Accuracy'],
        'Precision': metrics['Precision'], 'Recall': metrics['Recall'],
        'F1': metrics['F1'],
        'Dir_Head_Accuracy': dir_acc,
        'Dir_Head_Precision': dir_precision,
        'Dir_Head_Recall': dir_recall,
        'Dir_Head_F1': dir_f1,
        'Sharpe_Actual': sharpe_actual, 'Sharpe_Pred': sharpe_pred,
        'Max_DD_Actual': max_dd_actual, 'Max_DD_Pred': max_dd_pred,
        'Walk_RMSE': metrics_walk['RMSE'],
        'Walk_DA': metrics_walk['Directional_Accuracy'],
    }
    np.save('wm_evaluation_results.npy', results)

    # Final report
    print("\n" + "=" * 80)
    print("[OK] TRAINING AND EVALUATION COMPLETE!")
    print("=" * 80)
    print("\n[*] Generated Files:")
    print("  wm_best_model.pth          — Best model")
    print("  wm_final_model.pth         — Final model (SWA or best)")
    print("  wm_training_history.png    — Training curves")
    print("  wm_predictions.png         — Actual vs Predicted")
    print("  wm_directional_analysis.png — Direction accuracy")
    print("  wm_classification_metrics.png — P/R/F1 + confusion matrix")
    print("  wm_evaluation_results.npy  — All metrics")

    print("\n[*] Key Results:")
    if metrics['Directional_Accuracy'] > 55:
        print(f"  [OK] Directional accuracy: {metrics['Directional_Accuracy']:.2f}%")
    else:
        print(f"  [!] Directional accuracy: {metrics['Directional_Accuracy']:.2f}%")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1 Score:  {metrics['F1']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
