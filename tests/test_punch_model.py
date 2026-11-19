import numpy as np
import torch

from boxing.epam_dataset import FLIP_PERM


def test_flip_permutation_is_an_involution():
    assert [FLIP_PERM[i] for i in FLIP_PERM] == list(range(17))


def test_focal_gamma_zero_matches_weighted_cross_entropy():
    from boxing.losses import FocalCrossEntropy

    torch.manual_seed(0)
    logits = torch.randn(32, 8)
    target = torch.randint(0, 8, (32,))
    weight = torch.linspace(0.5, 2.0, 8)
    focal = FocalCrossEntropy(weight=weight, gamma=0.0)(logits, target)
    ref = torch.nn.CrossEntropyLoss(weight=weight)(logits, target)
    torch.testing.assert_close(focal, ref)


def test_focal_downweights_easy_examples():
    from boxing.losses import FocalCrossEntropy

    logits = torch.tensor([[8.0, 0.0], [0.5, 0.0]])  # first is easy, second is hard
    target = torch.tensor([0, 0])
    ce = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    focal = FocalCrossEntropy(gamma=2.0)(logits, target)
    # Focal loss with gamma>0 must be far smaller than CE on the easy example and
    # the mean must sit well below plain CE.
    assert focal.item() < ce.mean().item() * 0.5
