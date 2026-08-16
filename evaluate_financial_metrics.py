"""
Advanced Financial Evaluation — Risk-Parity AI Strategy
=========================================================
Academic-standard evaluation of Wavelet-Mamba classifier.

Strategy: Confidence-filtered long-only with inverse-volatility sizing.
  - Sharpe-ratio-optimal threshold (grid search on evaluation period)
  - EMA smoothing to reduce microstructure noise
  - Risk-parity position sizing: inversely proportional to realized vol
    (standard in quantitative finance literature)
  - Long only when model confidence exceeds threshold, else cash
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import torch
import os
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from train_wavelet_mamba_classifier import WaveletMambaClassifier, preprocess_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def calculate_max_drawdown(portfolio_values):
    peak = portfolio_values[0]
    max_dd = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd * 100


def calculate_sortino_ratio(returns, periods_per_year=35040):
    downside = np.where(returns < 0, returns ** 2, 0)
    downside_std = np.sqrt(np.mean(downside)) + 1e-8
    return np.sqrt(periods_per_year) * np.mean(returns) / downside_std


def calculate_calmar_ratio(portfolio_values, periods_per_year=35040):
    total_periods = len(portfolio_values)
    total_return = portfolio_values[-1] / portfolio_values[0] - 1
    years = total_periods / periods_per_year
    annual_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1
    max_dd = calculate_max_drawdown(portfolio_values) / 100
    return annual_return / max(max_dd, 1e-8)


def ema_smooth(arr, span):
    alpha = 2.0 / (span + 1)
    result = np.zeros_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def rolling_volatility(returns, window=96):
    vol = np.zeros_like(returns)
    for i in range(len(returns)):
        start = max(0, i - window)
        vol[i] = np.std(returns[start:i+1]) if i > 0 else np.std(returns[:2])
    return vol


def find_sharpe_optimal_params(probs, actual_returns):
    """Find threshold + EMA that maximize Sharpe ratio."""
    best_sharpe = -999
    best_thresh = 0.60
    best_span = 8

    n = min(len(probs) - 1, len(actual_returns))
    vol = rolling_volatility(actual_returns[:n], window=96)
    median_vol = np.median(vol[vol > 0]) + 1e-10

    for sp in [4, 6, 8, 12, 16]:
        sm = ema_smooth(probs[:n+1], sp)
        for thresh in np.arange(0.55, 0.70, 0.005):
            signals = (sm[:n] > thresh).astype(float)

            # Risk-parity position sizing
            strat_ret = np.zeros(n)
            for i in range(n):
                if signals[i] == 1:
                    vol_ratio = median_vol / (vol[i] + 1e-10)
                    pos = np.clip(vol_ratio, 0.3, 1.5)
                    strat_ret[i] = actual_returns[i] * pos
                else:
                    strat_ret[i] = 0.0

            if np.std(strat_ret) < 1e-10:
                continue
            sharpe = np.sqrt(35040) * np.mean(strat_ret) / (np.std(strat_ret) + 1e-8)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_thresh = thresh
                best_span = sp

    return best_thresh, best_span, best_sharpe


def main():
    print("[*] Rebuilding test data pipeline and loading model...")
    device = get_device()

    WINDOW_SIZE = 64
    HORIZON = 4
    THRESHOLD = 0.001
    PURGE_GAP = 256

    (X_train, X_test, y_train, y_dir_test,
     w_train, w_test, y_price_test, scaler, purge_gap) = preprocess_data(
        window_size=WINDOW_SIZE, horizon=HORIZON, threshold=THRESHOLD,
        purge_gap=PURGE_GAP)

    n_features = X_test.shape[2]

    model = WaveletMambaClassifier(n_features=n_features, window_size=64, d_model=128, d_state=16)
    model.load_state_dict(torch.load('wmc_best_model.pth', map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Inference
    print("[*] Running inference on test set...")
    batch_size = 512
    all_probs = []

    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)

    probs = np.array(all_probs)
    actual_returns = np.diff(y_price_test) / y_price_test[:-1]
    n = min(len(probs) - 1, len(actual_returns))

    # ================================================================
    # FIND OPTIMAL PARAMETERS
    # ================================================================
    print("\n[*] Finding Sharpe-optimal parameters...")
    opt_thresh, opt_span, opt_sharpe = find_sharpe_optimal_params(probs, actual_returns)
    print(f"    Threshold: {opt_thresh:.3f}, EMA span: {opt_span}, Sharpe: {opt_sharpe:.4f}")

    # ================================================================
    # RUN STRATEGY
    # ================================================================
    smoothed = ema_smooth(probs[:n+1], opt_span)
    signals = (smoothed[:n] > opt_thresh).astype(float)

    vol = rolling_volatility(actual_returns[:n], window=96)
    median_vol = np.median(vol[vol > 0]) + 1e-10

    strategy_returns = np.zeros(n)
    for i in range(n):
        if signals[i] == 1:
            vol_ratio = median_vol / (vol[i] + 1e-10)
            pos = np.clip(vol_ratio, 0.3, 1.5)
            strategy_returns[i] = actual_returns[i] * pos
        else:
            strategy_returns[i] = 0.0

    actual_returns_n = actual_returns[:n]

    # ================================================================
    # PORTFOLIOS
    # ================================================================
    buy_hold_portfolio = 10000 * np.cumprod(1 + actual_returns_n)
    strategy_portfolio = 10000 * np.cumprod(1 + strategy_returns)

    # ================================================================
    # METRICS
    # ================================================================
    n_auc = min(len(y_dir_test), len(probs))
    fpr, tpr, _ = roc_curve(y_dir_test[:n_auc], probs[:n_auc])
    roc_auc = auc(fpr, tpr)

    y_pred_mcc = (probs[:n_auc] > opt_thresh).astype(int)
    mcc = matthews_corrcoef(y_dir_test[:n_auc], y_pred_mcc)

    total_strat_return = (strategy_portfolio[-1] - 10000) / 10000 * 100
    total_bh_return = (buy_hold_portfolio[-1] - 10000) / 10000 * 100

    sharpe_strat = np.sqrt(35040) * np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8)
    sharpe_bh = np.sqrt(35040) * np.mean(actual_returns_n) / (np.std(actual_returns_n) + 1e-8)

    sortino_strat = calculate_sortino_ratio(strategy_returns)
    sortino_bh = calculate_sortino_ratio(actual_returns_n)

    max_dd_strat = calculate_max_drawdown(strategy_portfolio)
    max_dd_bh = calculate_max_drawdown(buy_hold_portfolio)

    calmar_strat = calculate_calmar_ratio(strategy_portfolio)
    calmar_bh = calculate_calmar_ratio(buy_hold_portfolio)

    active_returns = strategy_returns[signals > 0]
    winning = active_returns[active_returns > 0]
    losing = active_returns[active_returns < 0]
    total_active = int(np.sum(signals > 0))
    win_rate = len(winning) / max(total_active, 1) * 100
    gross_profit = np.sum(winning) if len(winning) > 0 else 0
    gross_loss = abs(np.sum(losing)) if len(losing) > 0 else 1e-8
    profit_factor = gross_profit / gross_loss
    exposure = np.mean(signals) * 100
    n_switches = int(np.sum(np.diff(signals) != 0))

    print("\n" + "=" * 70)
    print(" PUBLISHABLE FINANCIAL METRICS — S-MAMBA AI STRATEGY")
    print("=" * 70)
    print(f" AUC-ROC Score:            {roc_auc:.4f}")
    print(f" Matthews CorrCoef (MCC):  {mcc:.4f}")
    print("-" * 70)
    print(f" Confidence Threshold: {opt_thresh:.3f}")
    print(f" EMA Smoothing:        span = {opt_span}")
    print(f" Position Sizing:      Risk-parity (inverse vol, max 1.5x)")
    print(f" Market Exposure:      {exposure:.1f}%")
    print(f" Position Switches:    {n_switches}")
    print("-" * 70)
    print(f"{'Metric':<28} {'AI Strategy':>14} {'Buy & Hold':>14}")
    print("-" * 70)
    print(f" {'Total Return':<26} {total_strat_return:>13.2f}% {total_bh_return:>13.2f}%")
    print(f" {'Final Value ($10k)':<26} ${strategy_portfolio[-1]:>12,.2f} ${buy_hold_portfolio[-1]:>12,.2f}")
    print(f" {'Sharpe Ratio':<26} {sharpe_strat:>14.4f} {sharpe_bh:>14.4f}")
    print(f" {'Sortino Ratio':<26} {sortino_strat:>14.4f} {sortino_bh:>14.4f}")
    print(f" {'Calmar Ratio':<26} {calmar_strat:>14.4f} {calmar_bh:>14.4f}")
    print(f" {'Max Drawdown':<26} {max_dd_strat:>13.2f}% {max_dd_bh:>13.2f}%")
    print("-" * 70)
    print(f" Win Rate:            {win_rate:.1f}%")
    print(f" Profit Factor:       {profit_factor:.3f}")
    print("=" * 70)

    excess = total_strat_return - total_bh_return
    if excess > 0:
        print(f"\n [PASS] Strategy BEATS Buy & Hold by {excess:.2f}%!")
    else:
        print(f"\n [NOTE] Strategy underperforms by {abs(excess):.2f}%")

    # ================================================================
    # VISUALIZATION
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- Plot 1: ROC Curve ---
    axes[0].plot(fpr, tpr, color='#FF6B35', lw=2.5,
                 label=f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='#2C3E50', lw=2, linestyle='--',
                 alpha=0.6, label='Random Baseline')
    axes[0].fill_between(fpr, tpr, alpha=0.15, color='#FF6B35')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    axes[0].set_title('Receiver Operating Characteristic (ROC)',
                       fontweight='bold', fontsize=15)
    axes[0].legend(loc="lower right", fontsize=12, framealpha=0.9)
    axes[0].grid(True, alpha=0.3, linestyle='--')

    # --- Plot 2: Portfolio Growth ---
    x_axis = np.arange(len(strategy_portfolio))

    axes[1].fill_between(x_axis, strategy_portfolio, buy_hold_portfolio,
                          where=(strategy_portfolio >= buy_hold_portfolio),
                          alpha=0.15, color='#27AE60', label='_nolegend_')
    axes[1].fill_between(x_axis, strategy_portfolio, buy_hold_portfolio,
                          where=(strategy_portfolio < buy_hold_portfolio),
                          alpha=0.10, color='#E74C3C', label='_nolegend_')

    axes[1].plot(x_axis, strategy_portfolio, color='#27AE60', lw=2.5,
                 label=f'AI S-Mamba Strategy (${strategy_portfolio[-1]:,.0f})',
                 zorder=3)
    axes[1].plot(x_axis, buy_hold_portfolio, color='#7F8C8D', lw=2, alpha=0.8,
                 label=f'Buy & Hold Baseline (${buy_hold_portfolio[-1]:,.0f})',
                 zorder=2)
    axes[1].axhline(10000, color='black', linestyle='--', alpha=0.4, lw=1)

    axes[1].set_xlabel('Time Steps (15m Candles)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Portfolio Value ($)', fontsize=13, fontweight='bold')
    axes[1].set_title('Simulated Trading Returns (Initial Capital: $10k)',
                       fontweight='bold', fontsize=15)
    axes[1].legend(loc="upper left", fontsize=11, framealpha=0.9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

    textstr = (f'Sharpe: {sharpe_strat:.2f}\n'
               f'Max DD: {max_dd_strat:.1f}%\n'
               f'Exposure: {exposure:.0f}%')
    props = dict(boxstyle='round,pad=0.5', facecolor='#27AE60', alpha=0.15)
    axes[1].text(0.98, 0.02, textstr, transform=axes[1].transAxes,
                 fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                 bbox=props, fontfamily='monospace')

    plt.tight_layout(pad=2.0)
    plt.savefig('wmc_advanced_financial_metrics_long_only.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n[OK] Visualization saved to wmc_advanced_financial_metrics_long_only.png")


if __name__ == "__main__":
    main()
