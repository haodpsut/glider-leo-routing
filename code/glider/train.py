"""Training entry point: supervised regression of GLIDER onto CA-Global cost-to-go.

Usage:
    python -m glider.train --config configs/smoke.yaml --out runs/smoke
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .config import TrainConfig, load_train_config
from .dataset import sample_scenario, training_pairs
from .features import EDGE_FEAT_DIM, NODE_FEAT_DIM
from .model import GLIDER


# Next-hop ranking-loss conditioning.
OBS_COST_CLAMP_MS = 150.0
NEXTHOP_TEMPERATURE = 5.0


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model(cfg: TrainConfig) -> GLIDER:
    return GLIDER(
        node_dim=NODE_FEAT_DIM,
        edge_dim=EDGE_FEAT_DIM,
        hidden=cfg.hidden,
        num_layers=cfg.num_layers,
        geo_dim=1,
        use_messages=cfg.use_messages,
    )


def _sample_to_tensors(sample, device):
    f = sample.feats
    t = lambda a: torch.from_numpy(a).to(device)
    return {
        "node_feat": t(f.node_feat),
        "edge_index": t(f.edge_index),
        "edge_feat": t(f.edge_feat),
        "src_idx": t(sample.src_idx),
        "dst_idx": t(sample.dst_idx),
        "geo": t(sample.geo),
        "target": t(sample.target_q),
        "nh_dst": t(sample.nh_dst),
        "nh_nbr": t(sample.nh_nbr),
        "nh_geo": t(sample.nh_geo),
        "nh_obs_c": t(sample.nh_obs_c),
        "nh_mask": t(sample.nh_mask),
        "nh_target": t(sample.nh_target),
    }


def _nexthop_loss(model, h, batch):
    """Listwise cross-entropy: CA-Global next hop should minimise obs_c + Q(nbr,dest).

    Scores neighbours by ``-(obs_c + Q(nbr, dest))`` and applies cross-entropy toward
    the CA-Global-chosen neighbour index, so the greedy decision imitates the oracle.
    """
    nh_nbr = batch["nh_nbr"]                # (M, K)
    if nh_nbr.numel() == 0:
        return h.new_zeros(())
    M, K = nh_nbr.shape
    dst = batch["nh_dst"].unsqueeze(1).expand(M, K).reshape(-1)  # (M*K,)
    nbr = nh_nbr.reshape(-1)
    geo = batch["nh_geo"].reshape(-1, 1)
    q = model.cost_to_go(h, nbr, dst, geo).view(M, K)           # (M, K)
    # Clamp the observed link cost so a saturated neighbour's overload penalty does
    # not dominate the softmax, and use a temperature so the cross-entropy is well
    # conditioned when neighbour scores differ by only a few milliseconds.
    obs = torch.clamp(batch["nh_obs_c"], max=OBS_COST_CLAMP_MS)
    scores = -(obs + q) / NEXTHOP_TEMPERATURE
    scores = scores.masked_fill(~batch["nh_mask"], float("-inf"))
    return torch.nn.functional.cross_entropy(scores, batch["nh_target"])


def train(cfg: TrainConfig, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    print(f"[glider] device={device} params={sum(p.numel() for p in model.parameters())}")
    history = []
    t0 = time.time()
    running = []
    for step in range(1, cfg.steps + 1):
        scenario = sample_scenario(cfg.scenario, rng)
        sample = training_pairs(scenario, cfg.scenario, rng, cfg.max_pairs_per_dest)
        if len(sample.src_idx) == 0:
            continue
        batch = _sample_to_tensors(sample, device)
        model.train()
        h = model.embed(batch["node_feat"], batch["edge_index"], batch["edge_feat"])
        pred = model.cost_to_go(h, batch["src_idx"], batch["dst_idx"], batch["geo"])
        reg_loss = loss_fn(pred, batch["target"])
        nh_loss = _nexthop_loss(model, h, batch)
        loss = cfg.reg_weight * reg_loss + cfg.nh_weight * nh_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        running.append((float(reg_loss.item()), float(nh_loss.item())))

        if step % cfg.log_every == 0:
            recent = running[-cfg.log_every:]
            reg_avg = float(np.mean([r for r, _ in recent]))
            nh_avg = float(np.mean([n for _, n in recent]))
            rate = step / (time.time() - t0)
            print(f"[glider] step {step}/{cfg.steps} reg={reg_avg:.3f} nh_ce={nh_avg:.3f} ({rate:.1f} it/s)")
            history.append({"step": step, "reg_loss": reg_avg, "nh_ce": nh_avg, "loss": reg_avg + nh_avg})

    ckpt_path = os.path.join(out_dir, "glider.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": {
                "hidden": cfg.hidden,
                "num_layers": cfg.num_layers,
                "use_messages": cfg.use_messages,
            },
        },
        ckpt_path,
    )
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[glider] saved checkpoint -> {ckpt_path}")
    return ckpt_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/run")
    ap.add_argument("--seed", type=int, default=None, help="override the training seed")
    ap.add_argument("--steps", type=int, default=None, help="override the step count")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_train_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.steps is not None:
        cfg.steps = args.steps
    if args.device is not None:
        cfg.device = args.device
    train(cfg, args.out)


if __name__ == "__main__":
    main()
