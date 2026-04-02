"""
Learnable Wavelet Scattering Transform — OPTIMIZED
====================================================
Batched convolution implementation — no Python for-loops over wavelets.
All wavelet filters are stacked and applied in a single conv1d call.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LearnableScatteringLayer(nn.Module):
    """
    Learnable first-order wavelet scattering with BATCHED convolution.
    All J*Q wavelets are applied in ONE conv1d call instead of a loop.
    """

    def __init__(self, J=3, Q=1, kernel_size=16):
        super().__init__()
        self.J = J
        self.Q = Q
        self.n_filters = J * Q
        self.kernel_size = kernel_size

        # Learnable wavelet parameters (center freq and bandwidth)
        n = self.n_filters
        xi_inits = []
        sigma_inits = []
        for j in range(J):
            for q in range(Q):
                xi_inits.append(math.log(math.pi / (2 ** (j + q / max(Q, 1)))))
                sigma_inits.append(math.log(max(2.0 * (2 ** j), 1.0)))

        self.log_xi = nn.Parameter(torch.tensor(xi_inits))        # (n_filters,)
        self.log_sigma = nn.Parameter(torch.tensor(sigma_inits))   # (n_filters,)

        # Learnable low-pass
        self.log_phi_sigma = nn.Parameter(torch.tensor(math.log(float(kernel_size) / 4.0)))

    def _build_filters(self, channels, device):
        """Build all wavelet filters as a single stacked tensor — NO LOOPS at runtime"""
        xi = torch.exp(self.log_xi)          # (n_filters,)
        sigma = torch.exp(self.log_sigma)    # (n_filters,)

        half = self.kernel_size // 2
        t = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
        t = t[:self.kernel_size]  # (kernel_size,)

        # Broadcast: (n_filters, kernel_size)
        t_2d = t.unsqueeze(0)           # (1, K)
        xi_2d = xi.unsqueeze(1)         # (n, 1)
        sigma_2d = sigma.unsqueeze(1)   # (n, 1)

        gaussian = torch.exp(-t_2d ** 2 / (2 * sigma_2d ** 2))  # (n, K)
        real_filters = gaussian * torch.cos(xi_2d * t_2d)        # (n, K)
        imag_filters = gaussian * torch.sin(xi_2d * t_2d)        # (n, K)

        # Normalize each filter
        norms = torch.sqrt(real_filters.pow(2).sum(-1, keepdim=True)
                           + imag_filters.pow(2).sum(-1, keepdim=True) + 1e-8)
        real_filters = real_filters / norms  # (n, K)
        imag_filters = imag_filters / norms  # (n, K)

        # Tile for grouped conv: each filter applied to all channels
        # Shape needed: (channels * n_filters, 1, K) with groups=channels
        # We interleave: filter 0 for ch 0, filter 0 for ch 1, ..., filter 1 for ch 0, ...
        # Actually for groups=channels, shape is (out_channels, 1, K) where
        # out_channels = channels * n_filters and groups = channels
        # Each group (channel) gets n_filters output channels
        real_w = real_filters.unsqueeze(1).repeat(channels, 1, 1)  # (ch*n, 1, K)
        imag_w = imag_filters.unsqueeze(1).repeat(channels, 1, 1)  # (ch*n, 1, K)

        # Low-pass filter
        phi_sigma = torch.exp(self.log_phi_sigma)
        phi = torch.exp(-t ** 2 / (2 * phi_sigma ** 2))
        phi = phi / (phi.sum() + 1e-8)
        phi_w = phi.unsqueeze(0).unsqueeze(0).expand(channels * self.n_filters, -1, -1)

        return real_w, imag_w, phi_w

    def forward(self, x):
        """
        x: (batch, channels, time)
        Returns: scattering_coeffs (batch, channels * n_filters, time)
        """
        batch, channels, time = x.shape
        device = x.device
        pad = self.kernel_size // 2
        n = self.n_filters

        real_w, imag_w, phi_w = self._build_filters(channels, device)

        # Repeat input for grouped conv: (batch, channels, time) needs to become
        # input with channels repeated n_filters times for the grouped conv
        # Actually, groups=channels means each group has 1 input channel and n_filters output channels
        # We need to expand x to have channels * n_filters groups
        # Simpler approach: repeat x along channel dim, use groups = channels * n_filters
        x_rep = x.repeat(1, n, 1)  # (batch, channels * n, time)

        # Convolution: real and imag parts
        x_real = F.conv1d(x_rep, real_w, padding=pad, groups=channels * n)[:, :, :time]
        x_imag = F.conv1d(x_rep, imag_w, padding=pad, groups=channels * n)[:, :, :time]

        # Modulus
        modulus = torch.sqrt(x_real ** 2 + x_imag ** 2 + 1e-8)

        # Smooth with low-pass
        scattering = F.conv1d(modulus, phi_w, padding=pad, groups=channels * n)[:, :, :time]

        return scattering


class WaveletScatteringNetwork(nn.Module):
    """
    Learnable Wavelet Scattering Network (First-Order, Batched).
    Enhanced with volume-weighted coefficients and temporal deltas.
    All operations are batched tensor ops — no Python for-loops in forward pass.
    """

    def __init__(self, in_channels, out_channels, J=3, Q=1, kernel_size=16):
        super().__init__()
        self.scatter1 = LearnableScatteringLayer(J=J, Q=Q, kernel_size=kernel_size)
        self.in_channels = in_channels
        self.n_scatter = in_channels * J * Q

        # Order 0 (passthrough) + Order 1 (scattering) + deltas + volume-weighted
        # x + s1 + s1_delta + s1_vol_weighted = 1 + 3 multiples of scatter channels
        total_channels = in_channels + self.n_scatter * 3

        self.projection = nn.Sequential(
            nn.Conv1d(total_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

        # Separate energy projection for classification gating
        self.energy_proj = nn.Sequential(
            nn.Conv1d(self.n_scatter, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x, volume_weights=None):
        """
        x: (batch, in_channels, time) -> (batch, out_channels, time)
        volume_weights: optional (batch, 1, time) — normalized volume per timestep
        Also returns wavelet_energy: (batch, out_channels) — global avg energy for gating
        """
        s1 = self.scatter1(x)  # (B, n_scatter, T)

        # Temporal deltas: first-order difference of scattering coefficients
        s1_delta = torch.zeros_like(s1)
        s1_delta[:, :, 1:] = s1[:, :, 1:] - s1[:, :, :-1]

        # Volume-weighted scattering
        if volume_weights is not None:
            # volume_weights: (B, 1, T) — broadcast across scatter channels
            s1_vol = s1 * volume_weights
        else:
            s1_vol = s1

        # Concatenate all: [original, scattering, deltas, volume-weighted]
        scattered = torch.cat([x, s1, s1_delta, s1_vol], dim=1)

        # Wavelet energy for classification gating (global average over time)
        wavelet_energy = self.energy_proj(s1).mean(dim=2)  # (B, out_channels)

        return self.projection(scattered), wavelet_energy


if __name__ == "__main__":
    print("=" * 60)
    print("Testing OPTIMIZED Wavelet Scattering Network")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    x = torch.randn(4, 6, 64).to(device)
    vol = torch.rand(4, 1, 64).to(device)
    model = WaveletScatteringNetwork(in_channels=6, out_channels=128, J=3, Q=1, kernel_size=16).to(device)
    out, energy = model(x, volume_weights=vol)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Energy: {energy.shape}")

    # Benchmark
    import time
    model.eval()
    x_big = torch.randn(256, 128, 64).to(device)
    vol_big = torch.rand(256, 1, 64).to(device)
    model_big = WaveletScatteringNetwork(in_channels=128, out_channels=128, J=3, Q=1, kernel_size=16).to(device)

    # Warmup
    for _ in range(3):
        _ = model_big(x_big, volume_weights=vol_big)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(10):
        _ = model_big(x_big, volume_weights=vol_big)
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) / 10
    print(f"\nBenchmark (batch=256, ch=128, T=64): {elapsed*1000:.1f} ms/forward")
    print("[OK] Test passed!")
