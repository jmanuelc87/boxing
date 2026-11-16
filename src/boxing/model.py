"""RT-DETRv2 style detector with a selectable MobileNet / EfficientNet backbone.

Architecture (Chua & Tan, 2025, arXiv:2509.01241, "RT-DETRv2 Explained in 8
Illustrations"):

    image -> CNN backbone (strides 8/16/32)
          -> hybrid encoder (AIFI transformer on the S5 map + CCFM cross-scale fusion)
          -> uncertainty-minimal query selection (top-k encoder features -> queries)
          -> deformable-attention decoder (RT-DETRv2: per-level sampling points,
             selectable `decoder_method`) with iterative box refinement
          -> class logits + normalised (cx, cy, w, h) boxes per query

Everything after the backbone is Hugging Face's `RTDetrV2ForObjectDetection`, so
the model is a regular `PreTrainedModel`: `save_pretrained`, `from_pretrained`,
`Trainer`, `RTDetrImageProcessor` and the Hungarian/VFL losses all work as usual.
Only the backbone is swapped for a `timm` feature extractor chosen by name.

Usage:
    model = build_model("mobilenet", num_labels=6, id2label={0: "jab", ...})
    out = model(pixel_values=x, labels=targets)   # out.loss, out.logits, out.pred_boxes
"""

from __future__ import annotations

from typing import Any, Literal, cast

import timm
import torch
from transformers import RTDetrV2Config, RTDetrV2ForObjectDetection, TimmBackboneConfig

BackboneFamily = Literal["mobilenet", "efficientnet"]

# Family -> default timm model. Any timm name that starts with the family prefix
# ("mobilenet*", "efficientnet*", "tf_efficientnet*") is also accepted verbatim.
BACKBONES: dict[str, str] = {
    "mobilenet": "mobilenetv3_large_100",
    "efficientnet": "efficientnet_b0",
}

# Indices into timm's feature pyramid (reductions 2, 4, 8, 16, 32) -> strides 8/16/32,
# the three levels RT-DETRv2's hybrid encoder consumes.
_OUT_INDICES = (2, 3, 4)


def resolve_backbone(backbone: str) -> str:
    """Map a family name ('mobilenet', 'efficientnet') or timm model name to a timm name."""
    if backbone in BACKBONES:
        return BACKBONES[backbone]
    if "mobilenet" in backbone or "efficientnet" in backbone:
        return backbone
    raise ValueError(
        f"Unsupported backbone {backbone!r}. Use one of {sorted(BACKBONES)} "
        "or a timm MobileNet/EfficientNet model name (e.g. 'efficientnet_b3')."
    )


def build_config(
    backbone: str = "mobilenet",
    num_labels: int = 80,
    id2label: dict[int, str] | None = None,
    num_queries: int = 100,
    **overrides,
) -> RTDetrV2Config:
    """Create an `RTDetrV2Config` whose backbone is a timm MobileNet/EfficientNet."""
    timm_name = resolve_backbone(backbone)

    # Probe the (randomly initialised) backbone for its channel/stride layout.
    probe: Any = timm.create_model(
        timm_name, pretrained=False, features_only=True, out_indices=_OUT_INDICES
    )
    channels = tuple(probe.feature_info.channels())
    strides = tuple(probe.feature_info.reduction())
    del probe

    backbone_config = TimmBackboneConfig(  # type: ignore[call-arg]
        backbone=timm_name,
        out_indices=list(_OUT_INDICES),
        features_only=True,
    )

    params: dict[str, Any] = dict(
        backbone_config=backbone_config,
        encoder_in_channels=channels,
        feat_strides=strides,
        num_queries=num_queries,
        # The upstream defaults are tuned for ResNet-50 on COCO; these suit a small
        # backbone and a handful of classes.
        encoder_hidden_dim=256,
        d_model=256,
        decoder_layers=4,
        num_denoising=50,
    )
    if id2label is not None:
        params["id2label"] = {int(k): v for k, v in id2label.items()}
        params["label2id"] = {v: int(k) for k, v in id2label.items()}
    else:
        params["id2label"] = {i: f"LABEL_{i}" for i in range(num_labels)}
        params["label2id"] = {v: k for k, v in params["id2label"].items()}
    params.update(overrides)
    return RTDetrV2Config(**params)


class BoxingDetector(RTDetrV2ForObjectDetection):
    """RT-DETRv2 detector with a timm MobileNet/EfficientNet backbone.

    A thin subclass so the architecture has its own name; weights, forward and
    loss are exactly those of `RTDetrV2ForObjectDetection`.
    """

    config_class = RTDetrV2Config

    def __init__(self, config: RTDetrV2Config):
        super().__init__(config)
        # transformers picks the loss from the class name; a subclass name would
        # fall back to the causal-LM loss, so pin the RT-DETR (VFL + Hungarian) one.
        self.loss_type = "RTDetrV2ForObjectDetection"

    def load_backbone_weights(self) -> None:
        """Load ImageNet weights from timm into the backbone."""
        timm_name = cast(
            str, cast(TimmBackboneConfig, self.config.backbone_config).backbone
        )
        pretrained = timm.create_model(
            timm_name, pretrained=True, features_only=True, out_indices=_OUT_INDICES
        )
        target = self.model.backbone.model._backbone
        missing, unexpected = target.load_state_dict(
            pretrained.state_dict(), strict=False
        )
        if missing or unexpected:
            raise RuntimeError(
                f"Backbone weight mismatch: missing={missing}, unexpected={unexpected}"
            )


def build_model(
    backbone: BackboneFamily | str = "mobilenet",
    num_labels: int = 80,
    id2label: dict[int, str] | None = None,
    pretrained_backbone: bool = True,
    **config_overrides,
) -> BoxingDetector:
    """Build the detector.

    Args:
        backbone: "mobilenet", "efficientnet", or a timm model name such as
            "mobilenetv3_small_100" / "efficientnet_b2".
        num_labels: number of object classes (ignored if `id2label` is given).
        id2label: optional class-id -> name mapping.
        pretrained_backbone: initialise the backbone from timm ImageNet weights.
        **config_overrides: forwarded to `RTDetrV2Config` (e.g. `num_queries=300`).
    """
    config = build_config(backbone, num_labels, id2label, **config_overrides)
    model = BoxingDetector(config)
    if pretrained_backbone:
        model.load_backbone_weights()
    return model
