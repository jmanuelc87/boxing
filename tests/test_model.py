import pytest
import torch

from boxing.model import BACKBONES, BoxingDetector, build_config, build_model, resolve_backbone

ID2LABEL = {0: "jab", 1: "hook", 2: "uppercut", 3: "cross"}


@pytest.mark.parametrize("name", ["mobilenet", "efficientnet"])
def test_resolve_family_name(name):
    assert resolve_backbone(name) == BACKBONES[name]


def test_resolve_accepts_timm_name():
    assert resolve_backbone("efficientnet_b2") == "efficientnet_b2"


def test_resolve_rejects_unknown_backbone():
    with pytest.raises(ValueError):
        resolve_backbone("resnet50")


def test_build_config_sets_labels_and_channels():
    config = build_config("mobilenet", id2label=ID2LABEL, num_queries=10)
    assert config.id2label == ID2LABEL
    assert config.label2id == {v: k for k, v in ID2LABEL.items()}
    assert config.num_queries == 10
    assert len(config.encoder_in_channels) == 3
    assert len(config.feat_strides) == 3


@pytest.mark.parametrize("name", ["mobilenet", "efficientnet"])
def test_forward_shapes(name):
    model = build_model(name, id2label=ID2LABEL, pretrained_backbone=False, num_queries=20).eval()
    assert isinstance(model, BoxingDetector)
    with torch.no_grad():
        out = model(pixel_values=torch.randn(1, 3, 320, 320))
    assert out.logits.shape == (1, 20, len(ID2LABEL))
    assert out.pred_boxes.shape == (1, 20, 4)


def test_training_loss_and_backward():
    model = build_model("mobilenet", id2label=ID2LABEL, pretrained_backbone=False, num_queries=20).train()
    labels = [
        {"class_labels": torch.tensor([0, 2]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.4, 0.1, 0.1]])},
    ]
    out = model(pixel_values=torch.randn(1, 3, 320, 320), labels=labels)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert any(p.grad is not None for p in model.model.backbone.parameters())


def test_save_and_reload_round_trip(tmp_path):
    model = build_model("mobilenet", id2label=ID2LABEL, pretrained_backbone=False, num_queries=20).eval()
    model.save_pretrained(tmp_path)
    reloaded = BoxingDetector.from_pretrained(tmp_path).eval()
    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        assert torch.allclose(model(pixel_values=x).logits, reloaded(pixel_values=x).logits, atol=1e-5)
    assert reloaded.config.id2label == ID2LABEL
