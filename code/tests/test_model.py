import numpy as np
import torch

from glider.dataset import ScenarioConfig, sample_scenario, training_pairs
from glider.features import EDGE_FEAT_DIM, NODE_FEAT_DIM
from glider.model import GLIDER


def _tiny_cfg():
    cfg = ScenarioConfig(presets=["tiny"], n_ground_stations=6, ca_iters=6, traffic_base_gbps=60.0)
    cfg.queue.queue_delay_scale = 50.0
    return cfg


def test_forward_shape_and_nonneg():
    cfg = _tiny_cfg()
    rng = np.random.default_rng(0)
    scen = sample_scenario(cfg, rng)
    sample = training_pairs(scen, cfg, rng, max_pairs_per_dest=16)
    model = GLIDER(NODE_FEAT_DIM, EDGE_FEAT_DIM, hidden=16, num_layers=2)
    pred = model(
        torch.from_numpy(sample.feats.node_feat),
        torch.from_numpy(sample.feats.edge_index),
        torch.from_numpy(sample.feats.edge_feat),
        torch.from_numpy(sample.src_idx),
        torch.from_numpy(sample.dst_idx),
        torch.from_numpy(sample.geo),
    )
    assert pred.shape == (len(sample.src_idx),)
    assert torch.all(pred >= 0)  # softplus output


def test_ablation_no_messages_runs():
    cfg = _tiny_cfg()
    rng = np.random.default_rng(1)
    scen = sample_scenario(cfg, rng)
    sample = training_pairs(scen, cfg, rng, max_pairs_per_dest=16)
    model = GLIDER(NODE_FEAT_DIM, EDGE_FEAT_DIM, hidden=16, num_layers=3, use_messages=False)
    h = model.embed(
        torch.from_numpy(sample.feats.node_feat),
        torch.from_numpy(sample.feats.edge_index),
        torch.from_numpy(sample.feats.edge_feat),
    )
    assert h.shape[0] == sample.feats.node_feat.shape[0]


def test_gradients_flow():
    cfg = _tiny_cfg()
    rng = np.random.default_rng(2)
    scen = sample_scenario(cfg, rng)
    sample = training_pairs(scen, cfg, rng, max_pairs_per_dest=16)
    model = GLIDER(NODE_FEAT_DIM, EDGE_FEAT_DIM, hidden=16, num_layers=2)
    pred = model(
        torch.from_numpy(sample.feats.node_feat),
        torch.from_numpy(sample.feats.edge_index),
        torch.from_numpy(sample.feats.edge_feat),
        torch.from_numpy(sample.src_idx),
        torch.from_numpy(sample.dst_idx),
        torch.from_numpy(sample.geo),
    )
    loss = torch.nn.functional.smooth_l1_loss(pred, torch.from_numpy(sample.target_q))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(torch.any(g != 0) for g in grads)
