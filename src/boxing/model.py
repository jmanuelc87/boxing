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

`BoxingDetector` also accepts a clip, `pixel_values` of shape
`(batch, seq_len, 3, H, W)` (what `boxing.dataset.collate_clips` produces),
instead of a single image. A causal, per-pixel LSTM (`TemporalFusion`) fuses
the clip's frames on the coarsest feature map before the unmodified hybrid
encoder/decoder/loss; build with `temporal=False` to skip it and treat a
clip's frames independently instead.

Usage:
    model = build_model("mobilenet", num_labels=6, id2label={0: "jab", ...})
    out = model(pixel_values=x, labels=targets)   # out.loss, out.logits, out.pred_boxes

    clip_out = model(pixel_values=clip_x, labels=clip_targets)  # clip_x: (B, T, 3, H, W)
"""

from __future__ import annotations

from typing import Any, Literal, cast

import timm
import torch
from torch import nn
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
    temporal: bool = True,
    **overrides,
) -> RTDetrV2Config:
    """Create an `RTDetrV2Config` whose backbone is a timm MobileNet/EfficientNet.

    `temporal` controls whether the model builds its clip-input LSTM (see
    `TemporalFusion`); it is stored on the config so saved models reload it.
    """
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
        temporal=temporal,
    )
    if id2label is not None:
        params["id2label"] = {int(k): v for k, v in id2label.items()}
        params["label2id"] = {v: int(k) for k, v in id2label.items()}
    else:
        params["id2label"] = {i: f"LABEL_{i}" for i in range(num_labels)}
        params["label2id"] = {v: k for k, v in params["id2label"].items()}
    params.update(overrides)
    return RTDetrV2Config(**params)


class TemporalFusion(nn.Module):
    """Causal, per-pixel LSTM across a clip's frames, added as a residual.

    Runs on the coarsest backbone/encoder feature map only (already the level
    RT-DETRv2's own AIFI self-attention targets by default), so the LSTM sees a
    small spatial grid (e.g. 20x20 at 640px input, stride 32) and is cheap even
    with many frames: every spatial position becomes an independent sequence of
    length `seq_len`, so a batch of `batch_size` clips is `batch_size * H * W`
    sequences fed to one `nn.LSTM` call.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # hidden_size == d_model (input size) so the output needs no projection
        # back before being added to the original feature map.
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)

    def forward(self, feat: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """`feat`: (batch_size * seq_len, C, H, W) -> same shape, with temporal context added."""
        bt, c, h, w = feat.shape
        if bt != batch_size * seq_len:
            raise ValueError(f"expected {batch_size * seq_len} frames, got {bt}")
        # (B, T, C, H, W) -> (B, H, W, T, C) -> (B*H*W, T, C): one LSTM sequence per pixel.
        x = feat.view(batch_size, seq_len, c, h, w).permute(0, 3, 4, 1, 2).reshape(-1, seq_len, c)
        out, _ = self.lstm(x)
        out = out.reshape(batch_size, h, w, seq_len, c).permute(0, 3, 4, 1, 2).reshape(bt, c, h, w)
        return feat + out


class BoxingDetector(RTDetrV2ForObjectDetection):
    """RT-DETRv2 detector with a timm MobileNet/EfficientNet backbone.

    A thin subclass so the architecture has its own name; weights, forward and
    loss are exactly those of `RTDetrV2ForObjectDetection`, with one addition:
    `pixel_values` may be a clip, `(batch, seq_len, 3, H, W)`, instead of a
    single image, `(batch, 3, H, W)`. `labels` for a clip is a list of
    `batch` lists of `seq_len` per-frame target dicts (what
    `boxing.dataset.collate_clips` produces, unflattened). A clip is run
    through `TemporalFusion` (see above) before the unmodified RT-DETRv2
    hybrid encoder/decoder/loss; a single image takes the plain, unchanged
    path and is unaffected by this class at all.
    """

    config_class = RTDetrV2Config

    def __init__(self, config: RTDetrV2Config):
        super().__init__(config)
        # transformers picks the loss from the class name; a subclass name would
        # fall back to the causal-LM loss, so pin the RT-DETR (VFL + Hungarian) one.
        self.loss_type = "RTDetrV2ForObjectDetection"
        # getattr(..., True): a config saved before `temporal` existed still loads.
        self.temporal_fusion = TemporalFusion(config.d_model) if getattr(config, "temporal", True) else None

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor | None = None, labels=None, **kwargs):
        if pixel_values.dim() == 4:  # single image per batch element: unchanged RT-DETRv2 path
            return super().forward(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels, **kwargs)

        batch_size, seq_len, c, h, w = pixel_values.shape
        flat_pixels = pixel_values.reshape(batch_size * seq_len, c, h, w)
        flat_labels = [frame for clip in labels for frame in clip] if labels is not None else None

        if self.temporal_fusion is None:
            return super().forward(pixel_values=flat_pixels, labels=flat_labels, **kwargs)

        # Replicate RTDetrV2Model.forward's own backbone + projection call (the
        # default pixel_mask it would build when none is given), fuse the clip's
        # frames on the coarsest level, then run the (otherwise untouched)
        # hybrid encoder ourselves. `RTDetrV2Model.forward`'s `inputs_embeds`
        # parameter cannot carry this multiscale list -- it does
        # `inputs_embeds.shape[0]`, which only works for a single tensor, even
        # though the encoder it would feed indexes it as a list -- so instead
        # we hand it the finished `encoder_outputs` directly, which it accepts
        # as is. `pixel_values` must still be passed for `RTDetrV2Model.forward`
        # to recover `batch_size`/`device`; it recomputes (and discards) a
        # second, ungraded backbone pass over it, since `encoder_outputs` being
        # set skips calling the encoder again. That duplicate pass is wasted
        # compute, bounded by this project's small backbone, not a correctness
        # issue -- no gradient flows through a value nothing downstream uses.
        mask = torch.ones((batch_size * seq_len, h, w), device=flat_pixels.device)
        features = self.model.backbone(flat_pixels, mask)
        proj_feats = [self.model.encoder_input_proj[i](source) for i, (source, _) in enumerate(features)]
        proj_feats[-1] = self.temporal_fusion(proj_feats[-1], batch_size, seq_len)
        encoder_outputs = self.model.encoder(proj_feats)
        return super().forward(
            pixel_values=flat_pixels, pixel_mask=mask, encoder_outputs=encoder_outputs, labels=flat_labels, **kwargs
        )

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
    temporal: bool = True,
    **config_overrides,
) -> BoxingDetector:
    """Build the detector.

    Args:
        backbone: "mobilenet", "efficientnet", or a timm model name such as
            "mobilenetv3_small_100" / "efficientnet_b2".
        num_labels: number of object classes (ignored if `id2label` is given).
        id2label: optional class-id -> name mapping.
        pretrained_backbone: initialise the backbone from timm ImageNet weights.
        temporal: build the clip-input LSTM (`TemporalFusion`). When False, a
            clip input is still accepted but its frames are treated
            independently, same as `boxing.dataset.flatten_clips`.
        **config_overrides: forwarded to `RTDetrV2Config` (e.g. `num_queries=300`).
    """
    config = build_config(backbone, num_labels, id2label, temporal=temporal, **config_overrides)
    model = BoxingDetector(config)
    if pretrained_backbone:
        model.load_backbone_weights()
    return model
