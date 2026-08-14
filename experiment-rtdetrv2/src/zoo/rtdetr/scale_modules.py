"""
scale_modules.py  (RT-DETRv2 port)
----------------------------------
Lesion-adapted pre-decoder enhancement modules for RT-DETRv2 (official repo).

Identical modules to the Ultralytics version (DSA, FAR/FFT, SARE, ContextFusion,
LesionSCALE). They are framework-agnostic nn.Modules: input/output is a list of
the three multi-scale feature maps {P3, P4, P5} (each 256-ch) that the
HybridEncoder produces and the decoder consumes.

CHANGE vs the RT-DETR-L experiments: GAMMA_INIT = 0.1 (was 1.0).
  Lesson learned on RT-DETR-L:
    - gamma = 1.0 -> modules active but DISRUPTED the COCO-pretrained features
                     (hurt performance).
    - gamma = 0.0 -> modules dormant (internal weights get zero gradient, never
                     learn).
  A small positive init (0.1) starts each branch ~ identity (gentle on the
  pretrained weights) while STILL giving the module + gate real gradients, so
  the modules actually learn and self-scale. Best honest shot for SCALE to help.

Author: Praveen Kumar Murali | June 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Learnable residual gate init (y = x + gamma * module(x)). Small positive value:
# gentle start on pretrained weights, but non-zero gradient so modules train.
GAMMA_INIT = 0.1


# ──────────────────────────────────────────────────────────────────────────────
# 1. Dynamic Scale Aggregation  (cross-scale interaction)
# ──────────────────────────────────────────────────────────────────────────────
class DynamicScaleAggregation(nn.Module):
    """Share context across the 3 feature scales via attention over scale tokens."""

    def __init__(self, channels: int = 256, num_scales: int = 3, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_scales = num_scales

        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.gamma_ctx = nn.Parameter(torch.full((1,), GAMMA_INIT))

        self.lateral = nn.ModuleList(
            nn.Conv2d(channels, channels, 1) for _ in range(num_scales)
        )
        self.gamma_spatial = nn.Parameter(torch.full((1,), GAMMA_INIT))

    def forward(self, feats):
        assert len(feats) == self.num_scales, "expected 3 scale features"
        B = feats[0].shape[0]

        tokens = torch.stack([f.mean(dim=(2, 3)) for f in feats], dim=1)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)

        out = []
        for l, f in enumerate(feats):
            ctx = tokens[:, l, :].view(B, self.channels, 1, 1)
            out.append(f + self.gamma_ctx * ctx)

        refined = [self.lateral[l](out[l]) for l in range(self.num_scales)]
        for l in range(self.num_scales - 2, -1, -1):
            up = F.interpolate(refined[l + 1], size=refined[l].shape[2:], mode="nearest")
            refined[l] = refined[l] + up
        for l in range(1, self.num_scales):
            down = F.interpolate(refined[l - 1], size=refined[l].shape[2:], mode="nearest")
            refined[l] = refined[l] + down

        return [out[l] + self.gamma_spatial * refined[l]
                for l in range(self.num_scales)]


# ──────────────────────────────────────────────────────────────────────────────
# 2. Frequency-Aware Reconstruction  (deepest scale only; FFT variant)
# ──────────────────────────────────────────────────────────────────────────────
class FrequencyAwareReconstruction(nn.Module):
    """Frequency-domain enhancement via 2D FFT + learnable spectral filter."""

    def __init__(self, channels: int = 256):
        super().__init__()
        self.freq_filter = nn.Conv2d(2 * channels, 2 * channels, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.gamma_freq = nn.Parameter(torch.full((1,), GAMMA_INIT))
        self.gamma_mlp = nn.Parameter(torch.full((1,), GAMMA_INIT))

    def forward(self, x):
        B, C, H, W = x.shape
        orig_dtype = x.dtype
        dev = "cuda" if x.is_cuda else "cpu"
        with torch.autocast(device_type=dev, enabled=False):
            xf = torch.fft.rfft2(x.float(), norm="ortho")
            stacked = torch.cat([xf.real, xf.imag], dim=1)
            stacked = F.conv2d(
                stacked,
                self.freq_filter.weight.float(),
                self.freq_filter.bias.float() if self.freq_filter.bias is not None else None,
            )
            real, imag = stacked.chunk(2, dim=1)
            xfc = torch.complex(real, imag)
            x_freq = torch.fft.irfft2(xfc, s=(H, W), norm="ortho")

        x_freq = x_freq.to(orig_dtype)
        x = x + self.gamma_freq * x_freq
        x = x + self.gamma_mlp * self.mlp(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# 3. Scale-Aware Residual Enhancement  (all scales)
# ──────────────────────────────────────────────────────────────────────────────
class ScaleAwareResidualEnhancement(nn.Module):
    """Direction-aware recalibration via strip pooling along H and W."""

    def __init__(self, channels: int = 256, kernel_size: int = 7, groups: int = 16):
        super().__init__()
        pad = kernel_size // 2
        self.conv_h = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.conv_w = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.gn_h = nn.GroupNorm(groups, channels)
        self.gn_w = nn.GroupNorm(groups, channels)
        self.gamma_e = nn.Parameter(torch.full((1,), GAMMA_INIT))

    def forward(self, x):
        B, C, H, W = x.shape
        p_h = x.mean(dim=3)
        a_h = torch.sigmoid(self.gn_h(self.conv_h(p_h)))
        p_w = x.mean(dim=2)
        a_w = torch.sigmoid(self.gn_w(self.conv_w(p_w)))
        attn = a_h.unsqueeze(3) * a_w.unsqueeze(2)
        return x + self.gamma_e * (x * attn)


# ──────────────────────────────────────────────────────────────────────────────
# 2b. Context Fusion (paper-faithful CF block)  — deepest scale only
# ──────────────────────────────────────────────────────────────────────────────
class ContextFusion(nn.Module):
    """Faithful version of the reference paper's Context Fusion (CFBlock)."""

    def __init__(self, channels: int = 256):
        super().__init__()
        self.acf = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )
        self.gamma1 = nn.Parameter(torch.full((1,), GAMMA_INIT))
        self.gamma2 = nn.Parameter(torch.full((1,), GAMMA_INIT))

    def forward(self, x):
        x = x + self.gamma1 * self.acf(x)
        x = x + self.gamma2 * self.mlp(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Full SCALE block: DSA -> CF/FAR (deepest) -> SARE
# ──────────────────────────────────────────────────────────────────────────────
class LesionSCALE(nn.Module):
    """Chain the three branches. Input/output: list of 3 feature maps {P3,P4,P5}."""

    def __init__(self, channels: int = 256, num_scales: int = 3,
                 use_dsa: bool = True, use_far: bool = True, use_sare: bool = True,
                 deep_type: str = "cf"):
        super().__init__()
        self.num_scales = num_scales
        self.use_dsa = use_dsa
        self.use_far = use_far
        self.use_sare = use_sare

        if use_dsa:
            self.dsa = DynamicScaleAggregation(channels, num_scales)
        if use_far:
            self.far = (ContextFusion(channels) if deep_type == "cf"
                        else FrequencyAwareReconstruction(channels))
        if use_sare:
            self.sare = nn.ModuleList(
                ScaleAwareResidualEnhancement(channels) for _ in range(num_scales)
            )

    def forward(self, feats):
        feats = list(feats)
        if self.use_dsa:
            feats = self.dsa(feats)
        if self.use_far:
            feats[-1] = self.far(feats[-1])     # deepest scale only
        if self.use_sare:
            feats = [self.sare[l](feats[l]) for l in range(self.num_scales)]
        return feats
