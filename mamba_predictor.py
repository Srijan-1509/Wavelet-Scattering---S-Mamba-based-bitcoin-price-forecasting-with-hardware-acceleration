import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from wavelet_scattering import WaveletScatteringNetwork




class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) — core of Mamba.
    Input-dependent discretization makes it selective.
    Enhanced with regime-conditioned delta for volatility adaptation.
    """

    def __init__(self, d_model, d_state=8, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand

        # Input projection (x -> z and x for SSM)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D convolution (causal, on the SSM branch)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )

        # A initialization (log-spaced, like S4)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Input-dependent projections for B, C, delta
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # Delta projection
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        with torch.no_grad():
            self.dt_proj.bias.fill_(math.log(0.01))

        # Regime conditioning: modulates delta based on volatility/ATR signal
        self.regime_proj = nn.Sequential(
            nn.Linear(1, self.d_inner),
            nn.Tanh(),
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x, regime_signal=None):
        """
        x: (B, T, d_model)
        regime_signal: optional (B, T, 1) — ATR/volatility for delta modulation
        """
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_conv = x_ssm.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_proj = self.x_proj(x_conv)
        B = x_proj[:, :, :self.d_state]
        C = x_proj[:, :, self.d_state:2 * self.d_state]
        dt_raw = x_proj[:, :, -1:]

        delta = F.softplus(self.dt_proj(dt_raw))

        # Regime conditioning: modulate delta with volatility signal
        if regime_signal is not None:
            # Adapt seq_len if regime_signal has different length (after pooling)
            if regime_signal.shape[1] != seq_len:
                regime_signal = F.interpolate(
                    regime_signal.transpose(1, 2), size=seq_len, mode='nearest'
                ).transpose(1, 2)
            regime_gate = 1.0 + self.regime_proj(regime_signal)  # (B, T, d_inner)
            delta = delta * regime_gate

        # Clamp delta to prevent numerical instability
        delta = delta.clamp(min=1e-6, max=5.0)

        A = -torch.exp(self.A_log)

        y = self._selective_scan(x_conv, delta, A, B, C)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv
        y = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan(self, x, delta, A, B, C):
        x = x.float()
        delta = delta.float()
        A = A.float()
        B = B.float()
        C = C.float()

        log_A_bar = delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)
        S = log_A_bar.cumsum(dim=1).clamp(-15, 15)  # prevent exp overflow
        exp_S = torch.exp(S)
        exp_neg_S = torch.exp(-S)
        term = B_bar * x.unsqueeze(-1) * exp_neg_S
        sum_term = term.cumsum(dim=1)
        h = exp_S * sum_term
        y = (h * C.unsqueeze(2)).sum(dim=-1)
        return y.to(x.dtype)


# ============================================================================
# CMBlock: Core Mamba block (as in CryptoMamba, but enhanced)
# ============================================================================

class CMBlock(nn.Module):
    """
    CMBlock: LayerNorm → Mamba SSM → Residual + Dropout
    Corresponds to CMBlock in CryptoMamba but with larger d_state.
    Enhanced with regime signal passthrough.
    """

    def __init__(self, d_model, d_state=8, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, regime_signal=None):
        residual = x
        x = self.norm(x)
        x = self.ssm(x, regime_signal=regime_signal)
        x = self.dropout(x)
        return x + residual


# ============================================================================
# C-Block: Hierarchical block containing multiple CMBlocks + MLP
# ============================================================================

class CBlock(nn.Module):
    """
    C-Block: Stack of CMBlocks followed by a sequence-length-adjusting MLP.
    Inspired by CryptoMamba's hierarchical C-Block structure, but augmented
    with wavelet features and deeper state representation.
    Enhanced with regime signal passthrough.
    """

    def __init__(self, d_model, n_cmblocks=2, d_state=8, target_seq_len=None, dropout=0.1):
        super().__init__()
        self.cmblocks = nn.ModuleList([
            CMBlock(d_model, d_state=d_state, d_conv=4, expand=2, dropout=dropout)
            for _ in range(n_cmblocks)
        ])
        self.target_seq_len = target_seq_len
        # MLP to adjust sequence length (like CryptoMamba)
        if target_seq_len is not None:
            self.seq_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )

    def forward(self, x, regime_signal=None):
        for block in self.cmblocks:
            x = block(x, regime_signal=regime_signal)

        if self.target_seq_len is not None and x.shape[1] != self.target_seq_len:
            # Adaptive pool along sequence dimension
            x = x.transpose(1, 2)  # (B, D, T)
            x = F.adaptive_avg_pool1d(x, self.target_seq_len)
            x = x.transpose(1, 2)  # (B, T_new, D)
            x = self.seq_mlp(x)

        return x


# ============================================================================
# Multi-Scale Temporal Attention Pooling (Novel)
# ============================================================================

class TemporalAttentionPool(nn.Module):
    """
    Attention-weighted pooling over sequence dimension.
    Combines last, mean, and max features with learned attention.
    Novel vs CryptoMamba which uses simple linear merge.
    """

    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        self.gate = nn.Linear(d_model * 3, d_model)

    def forward(self, x):
        # x: (B, T, D)
        # Attention-weighted mean
        attn_weights = self.attn(x)  # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        x_attn = (x * attn_weights).sum(dim=1)  # (B, D)

        # Last hidden state
        x_last = x[:, -1, :]  # (B, D)

        # Max pool
        x_max = x.max(dim=1).values  # (B, D)

        # Gated fusion
        fused = torch.cat([x_attn, x_last, x_max], dim=-1)  # (B, 3D)
        return self.gate(fused)  # (B, D)


# ============================================================================
# Learnable Positional Encoding
# ============================================================================

class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding — more flexible than sinusoidal for financial data."""

    def __init__(self, d_model, max_len=256):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ============================================================================
# Full Wavelet-Mamba Predictor (Enhanced)
# ============================================================================

class WaveletMambaPredictor(nn.Module):
    """
    Enhanced Wavelet-Mamba Predictor with CryptoMamba-inspired hierarchy.

    Architecture:
      1. Input Projection (n_features -> d_model)
      2. Positional Encoding (learnable)
      3. Wavelet Scattering (learnable multi-scale features)
      4. C-Block 1: 2× CMBlocks (d_state=8), pool to seq//2
      5. C-Block 2: 2× CMBlocks (d_state=8), pool to seq//4
      6. C-Block 3: 2× CMBlocks (d_state=8), keep seq
      7. Merge: linear combination of all C-Block outputs
      8. Temporal Attention Pooling
      9. Dense refinement
     10. Dual output heads (price + direction)
    """

    def __init__(self, n_features=9, window_size=64, d_model=128, d_state=8):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.d_model = d_model

        # Layer 1: Input Projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.05),
        )

        # Layer 2: Positional Encoding
        self.pos_enc = LearnablePositionalEncoding(d_model, max_len=window_size + 16)

        # Layer 3: Wavelet Scattering
        self.wavelet_scatter = WaveletScatteringNetwork(
            in_channels=d_model,
            out_channels=d_model,
            J=3, Q=1,
            kernel_size=min(16, window_size // 2)
        )

        # Layers 4-6: Three C-Blocks (CryptoMamba-style hierarchy)
        seq1 = max(window_size // 2, 8)
        seq2 = max(window_size // 4, 4)

        self.cblock1 = CBlock(d_model, n_cmblocks=2, d_state=d_state,
                              target_seq_len=seq1, dropout=0.1)
        self.cblock2 = CBlock(d_model, n_cmblocks=2, d_state=d_state,
                              target_seq_len=seq2, dropout=0.1)
        self.cblock3 = CBlock(d_model, n_cmblocks=2, d_state=d_state,
                              target_seq_len=None, dropout=0.1)  # keeps seq2

        # Layer 7: Merge — project all C-Block outputs to same dim
        self.merge_proj1 = nn.Linear(d_model, d_model)
        self.merge_proj2 = nn.Linear(d_model, d_model)
        self.merge_proj3 = nn.Linear(d_model, d_model)
        self.merge_norm = nn.LayerNorm(d_model)

        # Layer 8: Temporal Attention Pooling
        self.temporal_pool = TemporalAttentionPool(d_model)

        # Layer 9: Dense refinement
        self.dense = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.05),
        )

        # Layer 10: Output heads
        self.price_head = nn.Linear(32, 1)
        self.direction_head = nn.Linear(32, 1)

    def forward(self, x):
        """
        x: (batch, window_size, n_features)
        Returns: price_pred (batch, 1), direction_logit (batch, 1)
        """
        # Layer 1: Input projection
        x = self.input_proj(x)  # (B, T, d_model)

        # Layer 2: Positional encoding
        x = self.pos_enc(x)

        # Layer 3: Wavelet scattering (channel-first)
        x_t = x.transpose(1, 2)  # (B, d_model, T)
        x_t = self.wavelet_scatter(x_t)
        x = x_t.transpose(1, 2)  # (B, T, d_model)

        # Layer 4: C-Block 1 → pools to T/2
        out1 = self.cblock1(x)  # (B, T/2, d_model)

        # Layer 5: C-Block 2 → pools to T/4
        out2 = self.cblock2(out1)  # (B, T/4, d_model)

        # Layer 6: C-Block 3 → keeps T/4
        out3 = self.cblock3(out2)  # (B, T/4, d_model)

        # Layer 7: Merge — aggregate all C-Block outputs
        # Pool out1 to T/4 for merging
        out1_pooled = F.adaptive_avg_pool1d(
            out1.transpose(1, 2), out3.shape[1]).transpose(1, 2)

        merged = (self.merge_proj1(out1_pooled) +
                  self.merge_proj2(out2) +
                  self.merge_proj3(out3))
        merged = self.merge_norm(merged)  # (B, T/4, d_model)

        # Layer 8: Temporal attention pooling
        pooled = self.temporal_pool(merged)  # (B, d_model)

        # Layer 9: Dense refinement
        x_out = self.dense(pooled)  # (B, 32)

        # Layer 10: Output heads
        price = self.price_head(x_out)
        direction = self.direction_head(x_out)

        return price, direction

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Loss Functions
# ============================================================================

class CombinedLoss(nn.Module):
    """
    Combined loss: Huber (price) + BCE with label smoothing (direction)
    + Directional consistency penalty.
    """

    def __init__(self, price_weight=0.6, direction_weight=0.3,
                 directional_penalty=0.1, huber_delta=1.0, label_smoothing=0.05):
        super().__init__()
        self.price_weight = price_weight
        self.direction_weight = direction_weight
        self.directional_penalty = directional_penalty
        self.huber = nn.SmoothL1Loss(beta=huber_delta)
        self.label_smoothing = label_smoothing

    def forward(self, price_pred, direction_logit, price_true, direction_true):
        # Huber loss for price
        loss_price = self.huber(price_pred, price_true)

        # BCE with label smoothing
        dir_targets = direction_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        loss_dir = F.binary_cross_entropy_with_logits(direction_logit, dir_targets)

        # Directional consistency penalty
        price_direction = torch.sign(price_pred - price_true)
        pred_direction = torch.sign(direction_logit)
        inconsistency = (price_direction != pred_direction).float().mean()

        total = (self.price_weight * loss_price +
                 self.direction_weight * loss_dir +
                 self.directional_penalty * inconsistency)

        return total, loss_price, loss_dir


# ============================================================================
# Wrapper for backward compatibility
# ============================================================================

class WaveletMambaWrapper:
    def __init__(self, window_size=64, n_features=9):
        self.window_size = window_size
        self.n_features = n_features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def build_model(self):
        print("\n" + "=" * 80)
        print("BUILDING ENHANCED WAVELET-MAMBA MODEL (PyTorch + CUDA)")
        print("=" * 80)

        print(f"  Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        self.model = WaveletMambaPredictor(
            n_features=self.n_features,
            window_size=self.window_size,
            d_model=128,
            d_state=8,
        ).to(self.device)

        print(f"\n  Enhanced Wavelet-Mamba Architecture:")
        print(f"  Layer 1:   Input Projection ({self.n_features} -> 128)")
        print(f"  Layer 2:   Learnable Positional Encoding")
        print(f"  Layer 3:   Learnable Wavelet Scattering (J=3, Q=1)")
        print(f"  Layer 4:   C-Block 1 — 2× CMBlocks (d_state=8), pool T/2")
        print(f"  Layer 5:   C-Block 2 — 2× CMBlocks (d_state=8), pool T/4")
        print(f"  Layer 6:   C-Block 3 — 2× CMBlocks (d_state=8)")
        print(f"  Layer 7:   Merge (aggregate all C-Block outputs)")
        print(f"  Layer 8:   Temporal Attention Pooling")
        print(f"  Layer 9:   Dense Refinement (128 -> 64 -> 32)")
        print(f"  Layer 10:  Dual Output (Price + Direction)")
        print(f"\n  d_state: 8 | d_conv: 4 | expand: 2")
        print(f"  Total CMBlocks: 6 (3 C-Blocks × 2)")
        print(f"  Total trainable parameters: {self.model.count_parameters():,}")

        print("\n" + "=" * 80)
        return self.model

    def summary(self):
        print("\nMODEL SUMMARY")
        print(self.model)
        print(f"\nTotal parameters: {self.model.count_parameters():,}")


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Enhanced Wavelet-Mamba Predictor")
    print("=" * 80)

    wrapper = WaveletMambaWrapper(window_size=64, n_features=9)
    wrapper.build_model()
    wrapper.summary()

    device = wrapper.device
    dummy = torch.randn(4, 64, 9).to(device)
    price, direction = wrapper.model(dummy)
    print(f"\nInput shape: {dummy.shape}")
    print(f"Price output: {price.shape}")
    print(f"Direction output: {direction.shape}")
    print("[OK] Forward pass successful!")
