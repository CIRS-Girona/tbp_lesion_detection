"""
scale_rtdetr.py
---------------
Integrates the LesionSCALE enhancement module into Ultralytics RT-DETR.

HOW IT WORKS (and why this approach):
  Ultralytics' model.train() REBUILDS the model from its YAML config via
  get_model(). So injecting a module into a model *instance* would be lost the
  moment training starts. To survive the rebuild, we patch the RTDETRDecoder
  CLASS itself:

    - patched __init__      : after the original init, build self.scale = LesionSCALE
    - patched _get_encoder_input : apply self.scale to the projected multi-scale
                                   features, right after input_proj and before the
                                   features are flattened for the decoder.

  This is exactly the placement from the reference paper: "after multi-scale
  channel projection, before decoder processing." Because the patch is on the
  class, every rebuild (including inside train()) reconstructs the decoder WITH
  SCALE, and the SCALE weights are saved/loaded as part of the decoder.

  The base RTDETRDecoder name stays in the YAML, so Ultralytics' parse_model
  special-casing for RTDETRDecoder keeps working unchanged.

USAGE:
  import scale_rtdetr
  scale_rtdetr.enable_scale(use_dsa=True, use_far=True, use_sare=True)  # before building
  from ultralytics import RTDETR
  model = RTDETR("rtdetr-l.pt")   # decoder now contains LesionSCALE
  model.train(...)

  # For the plain baseline, simply DON'T import/enable scale_rtdetr.

Author: Praveen Kumar Murali | June 2026
"""

import torch
from ultralytics.nn.modules import RTDETRDecoder

from scale_modules import LesionSCALE

# Keep references to the originals so we can restore / call through.
_ORIG_INIT = RTDETRDecoder.__init__
_ORIG_GET_ENCODER_INPUT = RTDETRDecoder._get_encoder_input

# Global config consulted by the patched __init__ (lets us toggle for ablation).
_SCALE_CFG = {
    "enabled": False,
    "use_dsa": True,
    "use_far": True,
    "use_sare": True,
    "deep_type": "cf",   # "cf" = paper-faithful Context Fusion; "fft" = our FFT module
}


def _num_scales(decoder) -> int:
    """Number of feature levels = number of input projection branches."""
    try:
        return len(decoder.input_proj)
    except Exception:
        return int(getattr(decoder, "nl", 3))


def _channels(decoder) -> int:
    """Unified channel dim after input_proj (RT-DETR hidden_dim, default 256)."""
    return int(getattr(decoder, "hidden_dim", 256))


def _patched_init(self, *args, **kwargs):
    _ORIG_INIT(self, *args, **kwargs)
    if _SCALE_CFG["enabled"]:
        self.scale = LesionSCALE(
            channels=_channels(self),
            num_scales=_num_scales(self),
            use_dsa=_SCALE_CFG["use_dsa"],
            use_far=_SCALE_CFG["use_far"],
            use_sare=_SCALE_CFG["use_sare"],
            deep_type=_SCALE_CFG["deep_type"],
        )
    else:
        self.scale = None


def _patched_get_encoder_input(self, x):
    # Project each scale to the unified hidden dim.
    x = [self.input_proj[i](feat) for i, feat in enumerate(x)]

    # Enhance the projected multi-scale features with LesionSCALE.
    if getattr(self, "scale", None) is not None:
        x = self.scale(x)

    # Flatten + concat for the decoder.
    feats = []
    shapes = []
    for feat in x:
        h, w = feat.shape[2:]
        feats.append(feat.flatten(2).permute(0, 2, 1))
        shapes.append([h, w])
    feats = torch.cat(feats, 1)
    return feats, shapes


def enable_scale(use_dsa: bool = True, use_far: bool = True, use_sare: bool = True,
                 deep_type: str = "cf"):
    """Activate the SCALE patch. Call BEFORE building the RTDETR model."""
    _SCALE_CFG.update(enabled=True, use_dsa=use_dsa, use_far=use_far,
                      use_sare=use_sare, deep_type=deep_type)
    RTDETRDecoder.__init__ = _patched_init
    RTDETRDecoder._get_encoder_input = _patched_get_encoder_input
    print(f"[scale_rtdetr] SCALE enabled: dsa={use_dsa} far={use_far} "
          f"sare={use_sare} deep_type={deep_type}")


def disable_scale():
    """Restore the original RTDETRDecoder (plain baseline)."""
    _SCALE_CFG["enabled"] = False
    RTDETRDecoder.__init__ = _ORIG_INIT
    RTDETRDecoder._get_encoder_input = _ORIG_GET_ENCODER_INPUT
    print("[scale_rtdetr] SCALE disabled (original RTDETRDecoder restored)")
