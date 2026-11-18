import pytest
import torch

from boxing.model import EPAMNet, EPAMNetConfig
from boxing.pretrained import convert_key, load_official_backbone


def test_convert_key_maps_every_official_name():
    assert convert_key("conv1_s.conv.weight") == "stem.0.0.weight"
    assert convert_key("conv1_s.bn.running_var") == "stem.0.1.running_var"
    assert convert_key("conv5.bn.bias") == "conv5.0.1.bias"
    assert convert_key("layer1.0.conv1.conv.weight") == "stages.0.0.expand.0.weight"
    assert convert_key("layer3.4.conv2.bn.weight") == "stages.2.4.depthwise.1.weight"
    assert convert_key("layer2.1.conv3.conv.weight") == "stages.1.1.project.0.weight"
    assert convert_key("layer4.0.downsample.bn.bias") == "stages.3.0.shortcut.1.bias"
    assert convert_key("layer2.2.se_module.fc2.bias") == "stages.1.2.se.fc2.bias"
    with pytest.raises(KeyError):
        convert_key("cls_head.fc_cls.weight")


def _official_names(key):
    import re

    parts = {"expand": "conv1", "depthwise": "conv2", "project": "conv3", "shortcut": "downsample", "se": "se_module"}
    if m := re.fullmatch(r"stem\.0\.([01])\.(.+)", key):
        return [f"conv1_s.{'conv' if m[1] == '0' else 'bn'}.{m[2]}"]
    if m := re.fullmatch(r"conv5\.0\.([01])\.(.+)", key):
        return [f"conv5.{'conv' if m[1] == '0' else 'bn'}.{m[2]}"]
    if m := re.fullmatch(r"stages\.(\d+)\.(\d+)\.se\.(.+)", key):
        return [f"layer{int(m[1]) + 1}.{m[2]}.se_module.{m[3]}"]
    m = re.fullmatch(r"stages\.(\d+)\.(\d+)\.(\w+)\.([01])\.(.+)", key)
    return [f"layer{int(m[1]) + 1}.{m[2]}.{parts[m[3]]}.{'conv' if m[4] == '0' else 'bn'}.{m[5]}"]


def test_load_official_backbone_is_strict_and_duplicates_the_pose_stem(tmp_path):
    model = EPAMNet(EPAMNetConfig(num_classes=3))
    donor = EPAMNet(EPAMNetConfig(num_classes=3))
    state = {}
    for ours, value in donor.pose_backbone.state_dict().items():
        (official,) = _official_names(ours)
        if ours == "stem.0.0.weight":
            value = value[:, :17]  # the official stem has one person's 17 channels
        state[f"backbone.{official}"] = value
    state["cls_head.fc_cls.weight"] = torch.zeros(120, 216)
    path = tmp_path / "pose.pth"
    torch.save({"state_dict": state}, path)

    load_official_backbone(model.pose_backbone, path, players=2)
    loaded = model.pose_backbone.state_dict()
    official_stem = state["backbone.conv1_s.conv.weight"]
    torch.testing.assert_close(loaded["stem.0.0.weight"][:, :17], official_stem / 2)
    torch.testing.assert_close(loaded["stem.0.0.weight"][:, 17:], official_stem / 2)
    torch.testing.assert_close(loaded["conv5.0.0.weight"], donor.pose_backbone.state_dict()["conv5.0.0.weight"])

    del state["backbone.conv5.conv.weight"]
    torch.save({"state_dict": state}, path)
    with pytest.raises(RuntimeError):
        load_official_backbone(model.pose_backbone, path, players=2)
