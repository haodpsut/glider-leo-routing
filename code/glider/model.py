"""GLIDER model: inductive message-passing GNN producing a cost-to-go function.

The network embeds every node with ``num_layers`` rounds of mean-aggregated message
passing over the (bidirectional) snapshot graph, then predicts a non-negative
cost-to-go ``Q(v, d)`` for any (node, destination) pair from the two node embeddings
plus an explicit geometric term. Greedy forwarding then picks the neighbour
minimising ``edge_cost(u, v) + Q(v, d)``.

Everything is plain PyTorch (no torch-geometric) so it installs cleanly on the
RTX 4090 host and message passing is a couple of ``index_add_`` calls.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(sizes: list[int], out_act: nn.Module | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class MessagePassingLayer(nn.Module):
    """One round of edge-conditioned, mean-aggregated message passing."""

    def __init__(self, hidden: int, edge_dim: int):
        super().__init__()
        self.msg = _mlp([2 * hidden + edge_dim, hidden, hidden])
        self.upd = nn.GRUCell(hidden, hidden)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        src, tgt = edge_index[0], edge_index[1]
        msg_in = torch.cat([h[src], h[tgt], edge_feat], dim=1)
        m = self.msg(msg_in)                                   # (E, hidden)

        agg = torch.zeros_like(h)
        agg.index_add_(0, tgt, m)                              # sum messages at target
        deg = torch.zeros(h.size(0), device=h.device)
        deg.index_add_(0, tgt, torch.ones(tgt.size(0), device=h.device))
        agg = agg / deg.clamp(min=1.0).unsqueeze(1)            # mean aggregation

        return self.upd(agg, h)                                # gated update


class GLIDER(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden: int = 64,
        num_layers: int = 4,
        geo_dim: int = 1,
        use_messages: bool = True,
    ):
        super().__init__()
        self.use_messages = use_messages
        self.hidden = hidden
        self.node_encoder = _mlp([node_dim, hidden, hidden])
        self.layers = nn.ModuleList(
            [MessagePassingLayer(hidden, edge_dim) for _ in range(num_layers)]
        )
        # Destination-conditioned readout -> non-negative cost-to-go.
        self.readout = _mlp([2 * hidden + geo_dim, hidden, hidden, 1], out_act=nn.Softplus())

    def embed(self, node_feat: torch.Tensor, edge_index: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        """Return (N, hidden) node embeddings."""
        h = self.node_encoder(node_feat)
        if self.use_messages:
            for layer in self.layers:
                h = layer(h, edge_index, edge_feat)
        return h

    def cost_to_go(
        self,
        h: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        geo: torch.Tensor,
    ) -> torch.Tensor:
        """Predict Q(src, dst) for batched index pairs -> (len,) non-negative."""
        feat = torch.cat([h[src_idx], h[dst_idx], geo], dim=1)
        return self.readout(feat).squeeze(-1)

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        geo: torch.Tensor,
    ) -> torch.Tensor:
        h = self.embed(node_feat, edge_index, edge_feat)
        return self.cost_to_go(h, src_idx, dst_idx, geo)
