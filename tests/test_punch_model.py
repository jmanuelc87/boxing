import numpy as np
import torch

from boxing.model import PUNCH_FEATURE_DIM, PunchPairClassifier, stream_features
from boxing.pair_dataset import FLIP_PERM, augment_window, normalize_window


def _window(t=16, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.uniform(100, 500, size=(2, t, 17, 3)).astype(np.float32)
    w[..., 2] = rng.uniform(0.3, 1.0, size=(2, t, 17))
    return w


def test_normalize_ignores_translation_and_zoom():
    w = _window()
    moved = w.copy()
    moved[..., :2] = moved[..., :2] * 2.0 + 50.0
    np.testing.assert_allclose(normalize_window(w), normalize_window(moved), atol=1e-4)


def test_normalize_keeps_missing_people_at_zero():
    w = _window()
    w[1, 3:6] = 0.0
    assert not normalize_window(w)[1, 3:6].any()


def test_flip_permutation_is_an_involution():
    assert [FLIP_PERM[i] for i in FLIP_PERM] == list(range(17))


def test_augment_keeps_shape_and_is_seeded():
    x = normalize_window(_window())
    a = augment_window(x, np.random.default_rng(1))
    b = augment_window(x, np.random.default_rng(1))
    assert a.shape == x.shape
    np.testing.assert_array_equal(a, b)


def test_stream_features_shape_and_missing_opponent_is_finite():
    x = torch.from_numpy(normalize_window(_window()))[None]  # (1, 2, T, 17, 3)
    x[:, 1] = 0.0  # opponent absent
    feats = stream_features(x[:, 0], x[:, 1], 1.0)
    assert feats.shape == (1, 16, PUNCH_FEATURE_DIM)
    assert torch.isfinite(feats).all()


def test_classifier_forward_shape():
    model = PunchPairClassifier(num_classes=8, hidden=32, layers=1)
    x = torch.from_numpy(np.stack([normalize_window(_window(seed=s)) for s in range(4)]))  # (4, 2, T, 17, 3)
    out = model(x)
    assert out.shape == (4, 8)
    assert torch.isfinite(out).all()


def test_focal_gamma_zero_matches_weighted_cross_entropy():
    from boxing.train_punch_classifier import FocalCrossEntropy

    torch.manual_seed(0)
    logits = torch.randn(32, 8)
    target = torch.randint(0, 8, (32,))
    weight = torch.linspace(0.5, 2.0, 8)
    focal = FocalCrossEntropy(weight=weight, gamma=0.0)(logits, target)
    ref = torch.nn.CrossEntropyLoss(weight=weight)(logits, target)
    torch.testing.assert_close(focal, ref)


def test_focal_downweights_easy_examples():
    from boxing.train_punch_classifier import FocalCrossEntropy

    logits = torch.tensor([[8.0, 0.0], [0.5, 0.0]])  # first is easy, second is hard
    target = torch.tensor([0, 0])
    ce = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    focal = FocalCrossEntropy(gamma=2.0)(logits, target)
    # Focal loss with gamma>0 must be far smaller than CE on the easy example and
    # the mean must sit well below plain CE.
    assert focal.item() < ce.mean().item() * 0.5
