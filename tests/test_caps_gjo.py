import numpy as np
from feature_selection.caps_gjo import CapsGJOSelector
from utils.common import load_yaml


def test_caps_selection_deterministic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 20))
    y = (X[:, 0] + 0.1 * rng.normal(size=64) > 0).astype(int)
    cfg = load_yaml("configs/caps_gjo.yaml")
    sel = CapsGJOSelector(cfg)
    idx1 = sel.select_k(X, y, k=8)
    idx2 = sel.select_k(X, y, k=8)
    assert idx1 == idx2 and len(idx1) > 0
