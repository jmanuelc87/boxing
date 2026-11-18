"""EPAM-Net: pose-driven, attention-guided two-stream network for video action recognition.

Reimplementation of EPAM-Net (Nady et al., Neurocomputing 2025, arXiv:2408.05421) in plain
PyTorch. The reference code lives in ``ahmed-nady/Multimodal-Action-Recognition`` and depends on
``mmcv`` and ``mmaction``; this file has no such dependency.

    images     (B, T, 3, H, W)     RGB frames, normalised, e.g. T=16 and 224x224
    keypoints  (B, P, T, J, 3)     (x, y, score) per joint, x and y in [0, 1] of the same crop
                                   as ``images`` (x to the right, y down); P=2 boxers, J=17

    model = EPAMNet(EPAMNetConfig(num_classes=8))
    out = model(images, keypoints)         # EPAMOutput(rgb_logits, pose_logits)
    probs = out.probabilities(model.config.rgb_weight)   # late fusion of the two streams' scores

``EPAMNet`` is a Hugging Face ``PreTrainedModel`` and ``EPAMNetConfig`` a ``PreTrainedConfig``, so
``save_pretrained``, ``from_pretrained`` and ``push_to_hub`` work. The module imports nothing from
``boxing``, so it can be shipped to the Hub as remote code (``trust_remote_code=True``).

Streams
-------
* **Pose stream.** The keypoints are rendered as Gaussian heatmaps, one channel per joint per
  boxer, ``(B, P*J, T, 56, 56)`` (`keypoints_to_heatmaps`). A small X-ShiftNet (X3D-style
  inverted-residual blocks, 2D convolutions with a Temporal Shift Module, 3 stages) encodes them
  to ``(B, 216, T, 7, 7)``.
* **RGB stream.** An X-ShiftNet-M (4 stages) encodes the frames to ``(B, 432, T, 7, 7)``.
* **Pose-guided attention.** The pose features give a spatial map (channel max and mean, a 7x7
  convolution, sigmoid) and a temporal weight (spatial average, a 1-D convolution over time,
  sigmoid). Their product re-weights the RGB features: ``rgb + rgb * attention``. This points the
  RGB stream at the key frames and at the regions where the boxers are.
* **Heads and fusion.** Each stream has its own classifier (the attended RGB features and the
  pose features). Train both with a classification loss; at inference average their softmax
  scores.

Differences from the reference
------------------------------
The reference samples 48 pose frames and 16 RGB frames and picks every third pose feature to
match. Here the pose and RGB streams see the same T frames by default, and the pose features are
strided only when their frame count is a multiple of the RGB one. The reference tracks one
person's 17 joints. Here the two boxers' joints are separate channels (34 in all),
so the pose stream can tell them apart. Their order is by track id and carries no role. The Gaussian width is not given in the paper text,
so ``sigma`` defaults to 0.6, the value PoseC3D uses at 56x56. It can be given per joint (one
radius for each of the 17 COCO keypoints), e.g. a tighter radius for the face than for the hips.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedConfig, PreTrainedModel
from transformers import initialization as init

NUM_JOINTS = 17
NUM_PLAYERS = 2


def sigma_per_joint(sigma: float | list[float], joints: int = NUM_JOINTS) -> torch.Tensor:
    """Gaussian width of each joint as a ``(J, 1)`` column, so it broadcasts over the heatmap rows.

    ``sigma`` is either one width for every joint or a list with one width per joint.
    """
    values = [float(sigma)] * joints if isinstance(sigma, (int, float)) else [float(s) for s in sigma]
    if len(values) != joints:
        raise ValueError(f"sigma has {len(values)} values, expected 1 or {joints} (one per joint)")
    if min(values) <= 0:
        raise ValueError("sigma must be positive for every joint")
    return torch.tensor(values, dtype=torch.float32).reshape(joints, 1)


def keypoints_to_heatmaps(
    keypoints: torch.Tensor, size: int = 56, sigma: float | list[float] = 0.6, min_score: float = 0.1
) -> torch.Tensor:
    """Render keypoints as Gaussian heatmaps (paper Eq. 1), weighted by joint confidence.

    ``keypoints``: ``(B, P, T, J, 3)`` with x and y in [0, 1] of the crop. Returns
    ``(B, P*J, T, size, size)``. Joints with a score at or below ``min_score`` give an empty map.
    ``sigma`` is one radius for all joints or one per joint (length J).
    """
    b, p, t, j, _ = keypoints.shape
    x = keypoints[..., 0] * size
    y = keypoints[..., 1] * size
    score = keypoints[..., 2] * (keypoints[..., 2] > min_score)
    grid = torch.arange(size, device=keypoints.device, dtype=keypoints.dtype) + 0.5
    s = sigma_per_joint(sigma, j).to(device=keypoints.device, dtype=keypoints.dtype)  # (J, 1)
    gx = torch.exp(-((grid - x.unsqueeze(-1)) ** 2) / (2 * s**2))  # (B, P, T, J, size)
    gy = torch.exp(-((grid - y.unsqueeze(-1)) ** 2) / (2 * s**2))
    heat = score[..., None, None] * gy[..., :, None] * gx[..., None, :]  # (B, P, T, J, size, size)
    return heat.permute(0, 1, 3, 2, 4, 5).reshape(b, p * j, t, size, size)


class TemporalShift(nn.Module):
    """Shift a fraction of the channels one frame forward and one backward in time (TSM).

    Costs no parameters or FLOPs. Frames shifted past either end are zero-filled.
    """

    def __init__(self, fold_div: int = 8):
        super().__init__()
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T, H, W)
        fold = x.shape[1] // self.fold_div
        if fold == 0 or x.shape[2] < 2:
            return x
        forward_shift = F.pad(x[:, :fold, :-1], (0, 0, 0, 0, 1, 0))
        backward_shift = F.pad(x[:, fold : 2 * fold, 1:], (0, 0, 0, 0, 0, 1))
        return torch.cat([forward_shift, backward_shift, x[:, 2 * fold :]], dim=1)


class SqueezeExcite3d(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Conv3d(channels, hidden, 1)
        self.fc2 = nn.Conv3d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool3d(x, 1)
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(scale))))


_Size3 = int | tuple[int, int, int]


def _conv_bn(
    in_ch: int, out_ch: int, kernel: _Size3, stride: _Size3 = 1, padding: _Size3 = 0, groups: int = 1
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel, stride=stride, padding=padding, groups=groups, bias=False),
        nn.BatchNorm3d(out_ch),
    )


class ShiftBlock(nn.Module):
    """X3D inverted-residual block made 2D: temporal shift, 1x1 expand, 3x3 depthwise, 1x1 project.

    The shift is applied to the residual branch only, so the shortcut keeps the original frames.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int, expand: float, use_se: bool, shift_div: int = 8):
        super().__init__()
        inner = int(round(out_ch * expand))
        self.shift = TemporalShift(shift_div)
        self.expand = _conv_bn(in_ch, inner, 1)
        self.depthwise = _conv_bn(inner, inner, (1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), groups=inner)
        self.se = SqueezeExcite3d(inner) if use_se else nn.Identity()
        self.project = _conv_bn(inner, out_ch, 1)
        self.shortcut = (
            _conv_bn(in_ch, out_ch, 1, stride=(1, stride, stride)) if stride != 1 or in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.expand(self.shift(x)))
        y = F.silu(self.se(self.depthwise(y)))
        y = self.project(y)
        return F.relu(y + self.shortcut(x))


class XShiftNet(nn.Module):
    """X3D-style backbone with 2D convolutions and temporal shift ("X-ShiftNet").

    ``stage_blocks`` and ``stage_channels`` give the blocks and the output width of each stage. The
    first block of every stage halves the spatial size. A final 1x1 convolution widens the last
    stage by ``expand``, so the output has ``round(stage_channels[-1] * expand)`` channels. The stem
    has no temporal convolution and the frame count is never reduced.
    """

    def __init__(
        self,
        in_channels: int,
        stage_blocks: tuple[int, ...],
        stage_channels: tuple[int, ...],
        stem_stride: int,
        expand: float = 2.25,
    ):
        super().__init__()
        if len(stage_blocks) != len(stage_channels):
            raise ValueError("stage_blocks and stage_channels must have the same length")
        stem_ch = stage_channels[0]
        self.stem = nn.Sequential(
            _conv_bn(in_channels, stem_ch, (1, 3, 3), stride=(1, stem_stride, stem_stride), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
        )
        stages, in_ch = [], stem_ch
        for blocks, out_ch in zip(stage_blocks, stage_channels):
            layers = []
            for i in range(blocks):
                # X3D puts squeeze-and-excitation on every other block ("half" style).
                layers.append(ShiftBlock(in_ch, out_ch, stride=2 if i == 0 else 1, expand=expand, use_se=(i + 1) % 2 == 0))
                in_ch = out_ch
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)
        self.out_channels = int(round(stage_channels[-1] * expand))
        self.conv5 = nn.Sequential(_conv_bn(in_ch, self.out_channels, 1), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T, H, W) -> (B, C', T, H', W')
        return self.conv5(self.stages(self.stem(x)))


class PoseGuidedAttention(nn.Module):
    """Nested spatial and temporal attention computed from the pose features.

    Spatial: channel-wise max and mean, a 1x7x7 convolution, sigmoid, giving ``(B, 1, T, H, W)``.
    Temporal: the spatial map averaged over H and W, a 1-D convolution over time, sigmoid, giving
    ``(B, 1, T, 1, 1)``. The output is their product, in [0, 1].
    """

    def __init__(self, kernel: int = 7):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv3d(2, 1, (1, kernel, kernel), padding=(0, kernel // 2, kernel // 2), bias=False),
            nn.BatchNorm3d(1),
        )
        self.temporal = nn.Conv1d(1, 1, kernel, padding=kernel // 2, bias=False)

    def forward(self, pose_feats: torch.Tensor) -> torch.Tensor:  # (B, C, T, H, W)
        b, _, t, _, _ = pose_feats.shape
        pooled = torch.cat([pose_feats.amax(dim=1, keepdim=True), pose_feats.mean(dim=1, keepdim=True)], dim=1)
        spatial = torch.sigmoid(self.spatial(pooled))  # (B, 1, T, H, W)
        temporal = torch.sigmoid(self.temporal(spatial.mean(dim=(3, 4)).reshape(b, 1, t)))
        return spatial * temporal.reshape(b, 1, t, 1, 1)


class ClassifierHead(nn.Module):
    """Dropout, global average pooling and one linear layer."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(F.adaptive_avg_pool3d(x, 1).flatten(1)))


class EPAMOutput(NamedTuple):
    rgb_logits: torch.Tensor  # attended RGB stream, (B, num_classes)
    pose_logits: torch.Tensor  # pose stream, (B, num_classes)

    def probabilities(self, rgb_weight: float = 0.5) -> torch.Tensor:
        """Late fusion: weighted mean of the two streams' softmax scores."""
        return rgb_weight * self.rgb_logits.softmax(-1) + (1 - rgb_weight) * self.pose_logits.softmax(-1)


class EPAMNetConfig(PreTrainedConfig):
    """Configuration of :class:`EPAMNet`.

    Args:
        num_classes: number of punch classes. Class names go in ``id2label``.
        heatmap_size: side of the pose heatmaps in pixels (56 pairs with 224x224 RGB, both ending
            at a 7x7 feature map).
        sigma: Gaussian width of a joint in heatmap pixels: one value for all joints, or a list
            with one value per joint (17, in COCO order).
        min_score: joints at or below this confidence are left out of the heatmaps.
        rgb_blocks / pose_blocks: blocks per stage of each stream. The defaults are X3D-M (3, 5,
            11, 7) for RGB and its last three stages (5, 11, 7) for pose.
        rgb_channels / pose_channels: output width of each stage.
        dropout: dropout before each classifier.
        rgb_weight: weight of the RGB stream when fusing the two streams' scores at inference.
    """

    model_type = "epam-net"

    def __init__(
        self,
        num_classes: int = 8,
        heatmap_size: int = 56,
        sigma: float | list[float] = 0.6,
        min_score: float = 0.1,
        rgb_blocks: tuple[int, ...] = (3, 5, 11, 7),
        pose_blocks: tuple[int, ...] = (5, 11, 7),
        rgb_channels: tuple[int, ...] = (24, 48, 96, 192),
        pose_channels: tuple[int, ...] = (24, 48, 96),
        dropout: float = 0.5,
        rgb_weight: float = 0.5,
        **kwargs,
    ):
        self.num_classes = num_classes
        self.heatmap_size = heatmap_size
        # A per-joint list is kept as a list so the config stays plain JSON.
        self.sigma = list(sigma) if isinstance(sigma, (list, tuple)) else sigma
        self.min_score = min_score
        # JSON turns tuples into lists; keep them tuples so a reloaded config compares equal.
        self.rgb_blocks, self.pose_blocks = tuple(rgb_blocks), tuple(pose_blocks)
        self.rgb_channels, self.pose_channels = tuple(rgb_channels), tuple(pose_channels)
        self.dropout = dropout
        self.rgb_weight = rgb_weight
        super().__init__(**kwargs)


class EPAMNet(PreTrainedModel):
    """Two-stream RGB + pose network with pose-guided attention. See the module docstring."""

    config_class = EPAMNetConfig
    base_model_prefix = "epam_net"
    main_input_name = "images"

    def __init__(self, config: EPAMNetConfig):
        super().__init__(config)
        self.heatmap_size, self.sigma, self.min_score = config.heatmap_size, config.sigma, config.min_score
        self.rgb_backbone = XShiftNet(3, config.rgb_blocks, config.rgb_channels, stem_stride=2)
        self.pose_backbone = XShiftNet(NUM_PLAYERS * NUM_JOINTS, config.pose_blocks, config.pose_channels, stem_stride=1)
        self.attention = PoseGuidedAttention()
        self.rgb_head = ClassifierHead(self.rgb_backbone.out_channels, config.num_classes, config.dropout)
        self.pose_head = ClassifierHead(self.pose_backbone.out_channels, config.num_classes, config.dropout)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv3d, nn.Conv1d)):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm3d):
            init.ones_(module.weight)
            init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            init.normal_(module.weight, std=0.01)
            init.zeros_(module.bias)

    @staticmethod
    def _align_pose(pose_feats: torch.Tensor, rgb_feats: torch.Tensor) -> torch.Tensor:
        """Match the pose features to the RGB features in frames (by striding) and in space."""
        t_pose, t_rgb = pose_feats.shape[2], rgb_feats.shape[2]
        if t_pose != t_rgb:
            if t_pose % t_rgb:
                raise ValueError(f"pose frames ({t_pose}) must be a multiple of RGB frames ({t_rgb})")
            pose_feats = pose_feats[:, :, :: t_pose // t_rgb]
        if pose_feats.shape[3:] != rgb_feats.shape[3:]:
            pose_feats = F.adaptive_avg_pool3d(pose_feats, rgb_feats.shape[2:])
        return pose_feats

    def forward(self, images: torch.Tensor, keypoints: torch.Tensor) -> EPAMOutput:
        """``images``: (B, T, 3, H, W). ``keypoints``: (B, 2, T_pose, 17, 3), x and y in [0, 1]."""
        rgb_feats = self.rgb_backbone(images.permute(0, 2, 1, 3, 4))  # (B, C, T, H', W')
        heatmaps = keypoints_to_heatmaps(keypoints, self.heatmap_size, self.sigma, self.min_score)
        pose_feats = self.pose_backbone(heatmaps)

        attention = self.attention(self._align_pose(pose_feats, rgb_feats))  # (B, 1, T, H', W')
        guided = rgb_feats + rgb_feats * attention
        return EPAMOutput(self.rgb_head(guided), self.pose_head(pose_feats))


# Lets save_pretrained copy this file to the Hub and AutoModel load it with trust_remote_code=True.
EPAMNetConfig.register_for_auto_class()
EPAMNet.register_for_auto_class("AutoModel")
