"""End-to-end smoke test: train a few steps, then evaluate all methods.

Asserts that the whole pipeline runs and that the learned policy produces a valid,
finite routing that delivers most demand -- not that it beats the baselines (that
needs a full training run).
"""

import numpy as np
import torch

from glider.config import load_train_config
from glider.evaluate import evaluate_methods
from glider.train import build_model, resolve_device, train
import os


CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "smoke.yaml")


def test_train_and_evaluate(tmp_path):
    cfg = load_train_config(CONFIG)
    cfg.steps = 120
    ckpt = train(cfg, str(tmp_path / "smoke"))
    assert os.path.exists(ckpt)

    device = resolve_device("cpu")
    from glider.evaluate import load_model
    from glider.features import EDGE_FEAT_DIM, NODE_FEAT_DIM

    model = load_model(ckpt, NODE_FEAT_DIM, EDGE_FEAT_DIM, device)
    results = evaluate_methods(cfg.scenario, model, device, n_scenarios=4, seed=7)

    assert set(results.keys()) == {
        "sp", "deflect_local", "ca_global", "deflect_oracle", "glider",
    }
    for method, mets in results.items():
        assert len(mets) == 4
        for m in mets:
            assert 0.0 <= m.carried_fraction <= 1.0
            assert m.total_flows > 0

    def mean_carried(m):
        return float(np.mean([x.carried_fraction for x in results[m]]))

    assert mean_carried("sp") > 0.8
    assert mean_carried("ca_global") > 0.8
    # The learned policy routes inside the progress-restricted action space, so even
    # barely trained it must deliver: it cannot loop or wander off. This is far
    # stronger than the old unrestricted-greedy expectation and is the whole point of
    # the deflection design.
    assert mean_carried("glider") > 0.8
