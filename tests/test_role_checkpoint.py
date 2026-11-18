import numpy as np
import torch

from boxing.train_role_classifier import load_checkpoint


def test_loads_plain_checkpoint(tmp_path):
    torch.save({"model": {"w": torch.ones(2)}, "roles": ["red", "blue"], "seed": 0}, tmp_path / "a.pt")
    assert load_checkpoint(tmp_path / "a.pt")["roles"] == ["red", "blue"]


def test_loads_older_checkpoint_that_weights_only_rejects(tmp_path):
    # A numpy scalar is rejected by torch.load(weights_only=True) on PyTorch >= 2.6.
    torch.save({"model": {"w": torch.ones(2)}, "seed": np.int64(3)}, tmp_path / "old.pt")
    assert int(load_checkpoint(tmp_path / "old.pt")["seed"]) == 3
