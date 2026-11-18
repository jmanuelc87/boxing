"""Load the official EPAM-Net backbones (mmaction checkpoints) into :class:`boxing.model.EPAMNet`.

The weights come from ahmed-nady/Multimodal-Action-Recognition (the Google Drive links in its
README), e.g. ``ntu120_xsub_x3dTShiftD_I3Dhead_RGB_epoch_205.pth`` for the RGB stream and
``ntu120_xsub_x3dTShiftPose_double_shifted_chs_best_top1_acc_epoch_235.pth`` for the pose stream.

    model = EPAMNet(config)
    load_official_backbone(model.rgb_backbone, "path/to/rgb.pth")
    load_official_backbone(model.pose_backbone, "path/to/pose.pth", players=2)

Only the backbone is taken. The classification heads (120 NTU classes) and the pose-guided
attention are left as initialised. The keys are renamed from mmaction's names to ours, and the
load is strict, so a shape or key mismatch raises instead of being skipped.

The official pose stem takes one person's 17 joints. Here a clip has two boxers, so the stem
weights are copied for each boxer and divided by the number of boxers: one boxer alone gives the
same response as in the official network, and the response to two is their average.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import nn

_PART = {"conv1": "expand", "conv2": "depthwise", "conv3": "project", "downsample": "shortcut", "se_module": "se"}


def convert_key(key: str) -> str:
    """Official key (without the ``backbone.`` prefix) to the key of :class:`boxing.model.XShiftNet`."""
    if m := re.fullmatch(r"conv1_s\.(conv|bn)\.(.+)", key):
        return f"stem.0.{0 if m[1] == 'conv' else 1}.{m[2]}"
    if m := re.fullmatch(r"conv5\.(conv|bn)\.(.+)", key):
        return f"conv5.0.{0 if m[1] == 'conv' else 1}.{m[2]}"
    if m := re.fullmatch(r"layer(\d+)\.(\d+)\.(conv[123]|downsample)\.(conv|bn)\.(.+)", key):
        return f"stages.{int(m[1]) - 1}.{m[2]}.{_PART[m[3]]}.{0 if m[4] == 'conv' else 1}.{m[5]}"
    if m := re.fullmatch(r"layer(\d+)\.(\d+)\.se_module\.(fc[12])\.(.+)", key):
        return f"stages.{int(m[1]) - 1}.{m[2]}.se.{m[3]}.{m[4]}"
    raise KeyError(f"unexpected key in the official checkpoint: {key}")


def load_checkpoint(path: str | Path) -> dict[str, torch.Tensor]:
    """The ``state_dict`` of an mmaction checkpoint, read on the CPU."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001  mmcv checkpoints can hold objects the safe loader refuses
        ckpt = torch.load(path, map_location="cpu", weights_only=False)  # the file is the official download
    return ckpt.get("state_dict", ckpt)


def load_official_backbone(backbone: nn.Module, path: str | Path, players: int = 1) -> None:
    """Strictly load an official ``backbone.*`` state into ``backbone``.

    ``players`` is how many people the stem's input channels hold (``1`` for RGB and for the
    official pose stream, ``2`` for the two-boxer pose stream).
    """
    state = {}
    for key, value in load_checkpoint(path).items():
        if not key.startswith("backbone."):
            continue  # cls_head: 120 NTU classes, not used
        new = convert_key(key[len("backbone.") :])
        if new == "stem.0.0.weight" and players > 1:
            value = torch.cat([value] * players, dim=1) / players
        state[new] = value
    backbone.load_state_dict(state, strict=True)
