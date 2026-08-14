"""
scale_modules.py
----------------
Lesion-adapted pre-decoder enhancement modules for RT-DETR.

These are framework-agnostic PyTorch nn.Modules. They operate on the three
multi-scale feature maps {P3, P4, P5} that RT-DETR feeds to its decoder, and
return three enhanced maps of identical shape. Each module wraps its effect in a
LEARNABLE RESIDUAL gate (gamma, see GAMMA_INIT) so the network can scale each
branch up or down; gamma is learnable and self-corrects if a branch is unhelpful.

Three branches, applied in sequence (DSA -> FAR -> SARE):

  1. DynamicScaleAggregation (DSA)
       Cross-scale attention so small and large lesions share context.
       (adapts the "Cross" branch of the reference design)

  2. FrequencyAwareReconstruction (FAR)   [deepest scale only]
       GENUINE frequency-domain filtering via 2D FFT with a learnable filter,
       to recover low-contrast lesion detail against textured skin.
       (our novel upgrade: the reference used conv+MLP, NOT frequency)

  3. ScaleAwareResidualEnhancement (SARE)
       Strip pooling along height and width to sharpen irregular lesion
       borders and recalibrate per-direction response.
       (adapts the "ELA" branch of the reference design)

Design targets skin lesion detection in TBP: small, multi-size, low-contrast
against textured background, irregular borders. Default channel dim = 256
(RT-DETR-L hidden dim). Spatial sizes are inferred at runtime (size-agnostic),
so the same code works at 1024x1024 or any input resolution.

Author: Praveen Kumar Murali | June 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Initial value for the learnable residual gates (y = x + gamma * module(x)).
# Initialising gamma to a real value lets the modules engage and train.
# Learnable, so it self-corrects down if a branch is unhelpful.
GAMMA_INIT = 1.0


# 1. Dynamic Scale Aggregation  (cross-scale interaction)
class DynamicScaleAggregation(nn.Module):
    """Share context across the 3 feature scales via attention over scale tokens.

    Each scale is summarised by global average pooling into one token; multi-head
    self-attention mixes the 3 tokens; the resulting per-scale context vector is
    broadcast back and added with a learnable residual. A light top-down /
    bottom-up convolution pass then restores spatial detail.
    """

    def __init__(self, channels: int = 256, num_scales: int = 3, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_scales = num_scales

        # Attention over scale tokens (sequence length = num_scales)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

        self.gamma_ctx = nn.Parameter(torch.full((1,), GAMMA_INIT))

        # Lateral 1x1 convs for top-down / bottom-up spatial refinement
        self.lateral = nn.ModuleList(
            nn.Conv2d(channels, channels, 1) for _ in range(num_scales)
        )
        self.gamma_spatial = nn.Parameter(torch.full((1,), GAMMA_INIT))

    def forward(self, feats):
        assert len(feats) == self.num_scales, "expected 3 scale features"
        B = feats[0].shape[0]

        # Build scale tokens via global average pooling -> (B, num_scales, C)
        tokens = torch.stack([f.mean(dim=(2, 3)) for f in feats], dim=1)

        # Self-attention across scales, with residual + norm
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)                 # (B, S, C)

        # Inject per-scale context back into each feature map (learnable residual)
        out = []
        for l, f in enumerate(feats):
            ctx = tokens[:, l, :].view(B, self.channels, 1, 1)
            out.append(f + self.gamma_ctx * ctx)

        # Light top-down + bottom-up spatial pass (FPN-style), learnable residual
        refined = [self.lateral[l](out[l]) for l in range(self.num_scales)]
        # top-down: add coarser (upsampled) into finer
        for l in range(self.num_scales - 2, -1, -1):
            up = F.interpolate(refined[l + 1], size=refined[l].shape[2:],
                               mode="nearest")
            refined[l] = refined[l] + up
        # bottom-up: add finer (downsampled) into coarser
        for l in range(1, self.num_scales):
            down = F.interpolate(refined[l - 1], size=refined[l].shape[2:],
                                 mode="nearest")
            refined[l] = refined[l] + down

        return [out[l] + self.gamma_spatial * refined[l]
                for l in range(self.num_scales)]


# 2. Frequency-Aware Reconstruction  (deepest scale only)
class FrequencyAwareReconstruction(nn.Module):
    """Genuine frequency-domain enhancement via 2D FFT + learnable filter.

    The feature map is transformed with a real 2D FFT. Real and imaginary parts
    are stacked and passed through a 1x1 convolution (a learnable, size-agnostic
    spectral filter) plus a pointwise channel MLP, then inverse-transformed. This
    recovers low-contrast lesion detail that is hard to separate from textured
    skin in the spatial domain. Wrapped in a learnable residual (starts at 0).

    This is the novel differentiator from the reference paper, whose context
    fusion branch used only spatial convolution + MLP (no frequency component).
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        # Learnable spectral filter on stacked [real, imag] -> 2C channels
        self.freq_filter = nn.Conv2d(2 * channels, 2 * channels, 1)
        # Pointwise channel MLP refinement in the spatial domain
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

        # FFT must run in float32: complex-half (from AMP/mixed precision) is not
        # supported by torch.fft / torch.complex. Disable autocast for this block
        # so the FFT and its learnable spectral filter stay in float32.
        dev = "cuda" if x.is_cuda else "cpu"
        with torch.autocast(device_type=dev, enabled=False):
            xf = torch.fft.rfft2(x.float(), norm="ortho")      # complex64 (B,C,H,W//2+1)
            stacked = torch.cat([xf.real, xf.imag], dim=1)     # (B, 2C, H, W//2+1) float32
            # Cast the filter weights to float32 to match the FFT data under AMP.
            # The .float() cast is differentiable, so gradients still reach freq_filter.
            stacked = F.conv2d(
                stacked,
                self.freq_filter.weight.float(),
                self.freq_filter.bias.float() if self.freq_filter.bias is not None else None,
            )
            real, imag = stacked.chunk(2, dim=1)
            xfc = torch.complex(real, imag)                    # complex64
            x_freq = torch.fft.irfft2(xfc, s=(H, W), norm="ortho")  # (B,C,H,W) float32

        x_freq = x_freq.to(orig_dtype)
        x = x + self.gamma_freq * x_freq                       # freq residual
        x = x + self.gamma_mlp * self.mlp(x)                   # channel MLP residual
        return x


# 3. Scale-Aware Residual Enhancement  (all scales)
class ScaleAwareResidualEnhancement(nn.Module):
    """Direction-aware recalibration via strip pooling along H and W.

    Pooling along width gives a height profile; pooling along height gives a width
    profile. Each is passed through a 1D conv + group norm + sigmoid to produce
    directional attention weights, whose outer product recalibrates the feature.
    Sharpens irregular / elongated lesion borders. Learnable residual (starts 0).
    """

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

        # Height profile: average over width -> (B, C, H)
        p_h = x.mean(dim=3)
        a_h = torch.sigmoid(self.gn_h(self.conv_h(p_h)))       # (B, C, H)

        # Width profile: average over height -> (B, C, W)
        p_w = x.mean(dim=2)
        a_w = torch.sigmoid(self.gn_w(self.conv_w(p_w)))       # (B, C, W)

        # Outer product of directional weights -> (B, C, H, W)
        attn = a_h.unsqueeze(3) * a_w.unsqueeze(2)
        return x + self.gamma_e * (x * attn)


# 2b. Context Fusion (paper-faithful CF block)  — deepest scale only
class ContextFusion(nn.Module):
    """Faithful version of the reference paper's Context Fusion (CFBlock).

    Two residual substructures applied to the deepest feature:
      Z   = X + g1 * A_cf(X)     (convolutional attention mapping)
      out = Z + g2 * M(Z)        (channel MLP)
    No frequency component (unlike FrequencyAwareReconstruction). This matches the
    paper's design: convolutional attention + MLP for deep contextual modelling.
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        # Convolutional attention mapping A_cf: depthwise spatial context +
        # pointwise channel mixing.
        self.acf = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
        )
        # Channel MLP M (pointwise, expand-reduce).
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


# Full SCALE block: DSA -> CF/FAR (deepest) -> SARE
class LesionSCALE(nn.Module):
    """Chain the three branches. Input/output: list of 3 feature maps {P3,P4,P5}.

    Toggle individual branches (for the ablation study) via the use_* flags.
    """

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
            # Deepest-scale module. "cf" = paper-faithful Context Fusion (conv +
            # MLP); "fft" = our Frequency-Aware Reconstruction (FFT-based).
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
