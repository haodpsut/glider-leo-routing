"""YAML config loading and construction of typed config objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

from .dataset import ScenarioConfig
from .network import NetworkConfig
from .queueing import QueueConfig


@dataclass
class TrainConfig:
    run_name: str = "run"
    seed: int = 0
    device: str = "auto"
    steps: int = 2000
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    reg_weight: float = 0.1        # cost-to-go regression (shaping / auxiliary)
    nh_weight: float = 1.0         # next-hop imitation (drives the greedy decision)
    hidden: int = 64
    num_layers: int = 4
    use_messages: bool = True
    eval_every: int = 200
    eval_scenarios: int = 8
    max_pairs_per_dest: int = 64
    log_every: int = 20
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)


def _scenario_from_dict(d: dict[str, Any]) -> ScenarioConfig:
    d = dict(d or {})
    net = NetworkConfig(**d.pop("net", {}))
    queue = QueueConfig(**d.pop("queue", {}))
    return ScenarioConfig(net=net, queue=queue, **d)


def load_train_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    scenario = _scenario_from_dict(raw.pop("scenario", {}))
    return TrainConfig(scenario=scenario, **raw)
