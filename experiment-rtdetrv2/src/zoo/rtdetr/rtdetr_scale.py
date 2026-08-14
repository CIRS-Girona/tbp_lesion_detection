"""
rtdetr_scale.py
---------------
RTDETRSCALE: RT-DETRv2 with the LesionSCALE pre-decoder feature reconstruction
module inserted between the HybridEncoder and the decoder.

    backbone -> encoder -> SCALE -> decoder

This mirrors the reference paper's "pre-decoder feature reconstruction": SCALE
operates on the encoder's 3 multi-scale output maps (each 256-ch) before they
enter the decoder. It is registered as model 'RTDETRSCALE', so a config selects
it with `model: RTDETRSCALE`. Same __inject__ as RTDETR (backbone/encoder/decoder
are built from their own config blocks); the extra kwargs configure SCALE.

The backbone/encoder/decoder submodule NAMES match RTDETR, so a COCO-pretrained
RTDETR checkpoint loads cleanly via `-t` (only the new `scale.*` params are fresh,
plus the class heads which differ for our single class).

Author: Praveen Kumar Murali | June 2026
"""

import torch.nn as nn

from ...core import register
from .scale_modules import LesionSCALE

__all__ = ['RTDETRSCALE']


@register()
class RTDETRSCALE(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self,
                 backbone: nn.Module,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 scale_channels: int = 256,
                 scale_num_scales: int = 3,
                 use_dsa: bool = True,
                 use_far: bool = True,
                 use_sare: bool = True,
                 deep_type: str = 'cf'):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.scale = LesionSCALE(
            channels=scale_channels,
            num_scales=scale_num_scales,
            use_dsa=use_dsa,
            use_far=use_far,
            use_sare=use_sare,
            deep_type=deep_type,
        )
        print(f"[RTDETRSCALE] SCALE enabled: dsa={use_dsa} far={use_far} "
              f"sare={use_sare} deep_type={deep_type} channels={scale_channels}")

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)          # list of 3 maps [P3, P4, P5], 256-ch each
        x = self.scale(x)            # pre-decoder feature reconstruction
        x = self.decoder(x, targets)
        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
