import pytest
import torch

from boxing.model import (
    EPAMNet,
    EPAMNetConfig,
    EPAMOutput,
    PoseGuidedAttention,
    TemporalShift,
    keypoints_to_heatmaps,
)

SMALL = dict(
    num_classes=8,
    rgb_blocks=(1, 1, 1, 1),
    rgb_channels=(8, 16, 24, 32),
    pose_blocks=(1, 1, 1),
    pose_channels=(8, 16, 24),
)


def _keypoints(batch=2, frames=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    kp = torch.rand(batch, 2, frames, 17, 3, generator=g)
    kp[..., 2] = 0.5 + 0.5 * kp[..., 2]  # scores in [0.5, 1]
    return kp


def test_heatmap_peaks_at_the_keypoint_and_scales_with_score():
    kp = torch.zeros(1, 2, 3, 17, 3)
    kp[0, 1, 2, 5] = torch.tensor([(28 + 0.5) / 56, (14 + 0.5) / 56, 0.8])  # boxer 1, frame 2, joint 5
    heat = keypoints_to_heatmaps(kp, size=56)
    assert heat.shape == (1, 34, 3, 56, 56)
    channel = 17 + 5  # boxer 1's joints follow boxer 0's
    m = heat[0, channel, 2]
    assert m.argmax().item() == 14 * 56 + 28
    assert m.max().item() == pytest.approx(0.8)
    assert heat.sum() == pytest.approx(heat[0, channel, 2].sum().item())  # nothing else is drawn


def test_low_confidence_and_missing_joints_give_empty_maps():
    kp = _keypoints(batch=1, frames=2)
    kp[0, 0, 0, 3, 2] = 0.05  # below min_score
    kp[0, 1, 1] = 0.0  # opponent absent on frame 1
    heat = keypoints_to_heatmaps(kp, size=56, min_score=0.1)
    assert heat[0, 3, 0].abs().sum() == 0
    assert heat[0, 17:, 1].abs().sum() == 0
    assert heat[0, 0, 0].abs().sum() > 0


def test_temporal_shift_moves_channel_groups_in_time_and_zero_pads():
    x = torch.arange(1 * 16 * 5 * 2 * 2, dtype=torch.float32).reshape(1, 16, 5, 2, 2)
    y = TemporalShift(fold_div=8)(x)  # fold = 2 channels per direction
    assert y.shape == x.shape
    assert not y[:, :2, 0].any()
    torch.testing.assert_close(y[:, :2, 1:], x[:, :2, :-1])
    assert not y[:, 2:4, -1].any()
    torch.testing.assert_close(y[:, 2:4, :-1], x[:, 2:4, 1:])
    torch.testing.assert_close(y[:, 4:], x[:, 4:])
    assert sum(p.numel() for p in TemporalShift().parameters()) == 0


def test_attention_is_a_bounded_spatial_temporal_map():
    att = PoseGuidedAttention()
    out = att(torch.randn(2, 12, 6, 7, 7))
    assert out.shape == (2, 1, 6, 7, 7)
    assert out.min() >= 0 and out.max() <= 1


def test_default_sizes_match_the_reference_widths():
    model = EPAMNet(EPAMNetConfig())
    assert model.rgb_backbone.out_channels == 432
    assert model.pose_backbone.out_channels == 216


def test_forward_shapes_and_late_fusion():
    model = EPAMNet(EPAMNetConfig(**SMALL)).eval()
    images = torch.randn(2, 4, 3, 224, 224)
    with torch.no_grad():
        out = model(images, _keypoints())
    assert isinstance(out, EPAMOutput)
    assert out.rgb_logits.shape == out.pose_logits.shape == (2, 8)
    probs = out.probabilities()
    torch.testing.assert_close(probs.sum(-1), torch.ones(2))
    torch.testing.assert_close(out.probabilities(rgb_weight=1.0), out.rgb_logits.softmax(-1))
    torch.testing.assert_close(out.probabilities(rgb_weight=0.0), out.pose_logits.softmax(-1))


def test_rgb_loss_reaches_the_pose_stream_through_the_attention():
    model = EPAMNet(EPAMNetConfig(**SMALL)).train()
    out = model(torch.randn(2, 4, 3, 224, 224), _keypoints())
    out.rgb_logits.sum().backward()
    assert all(p.grad is not None for p in model.pose_backbone.parameters())
    assert all(p.grad is not None for p in model.attention.parameters())
    assert all(p.grad is None for p in model.pose_head.parameters())


def test_pose_frames_are_strided_to_match_rgb_frames():
    model = EPAMNet(EPAMNetConfig(**SMALL)).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 4, 3, 224, 224), _keypoints(batch=1, frames=8))
    assert out.rgb_logits.shape == (1, 8)
    with pytest.raises(ValueError):
        model(torch.randn(1, 4, 3, 224, 224), _keypoints(batch=1, frames=6))


def test_mismatched_spatial_sizes_are_pooled_to_the_rgb_map():
    model = EPAMNet(EPAMNetConfig(**SMALL)).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 4, 3, 112, 112), _keypoints(batch=1))  # RGB map 4x4, pose map 7x7
    assert out.rgb_logits.shape == (1, 8)


def test_save_and_reload_roundtrip(tmp_path):
    config = EPAMNetConfig(**SMALL, id2label={i: f"c{i}" for i in range(8)}, rgb_weight=0.7)
    model = EPAMNet(config).eval()
    model.save_pretrained(tmp_path)
    reloaded = EPAMNet.from_pretrained(tmp_path).eval()
    assert reloaded.config.rgb_blocks == (1, 1, 1, 1) and reloaded.config.rgb_weight == 0.7
    assert reloaded.config.id2label[3] == "c3"
    images, kp = torch.randn(1, 4, 3, 224, 224), _keypoints(batch=1)
    with torch.no_grad():
        torch.testing.assert_close(model(images, kp).rgb_logits, reloaded(images, kp).rgb_logits)
