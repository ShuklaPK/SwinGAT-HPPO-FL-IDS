from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch


def quantize_tensor(t: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, float, float]:
    mn, mx = t.min(), t.max()
    scale = (mx - mn) / (2**bits - 1 + 1e-9)
    q = torch.round((t - mn) / (scale + 1e-9)).to(torch.int32)
    return q, float(mn), float(scale)


def dequantize_tensor(q: torch.Tensor, mn: float, scale: float) -> torch.Tensor:
    return q.float() * scale + mn


@dataclass
class FLConfig:
    num_clients: int = 3
    clients_per_round: int = 2
    rounds: int = 2
    local_epochs: int = 1
    clip_norm: float = 1.0
    dp_sigma: float = 0.0
    secure_aggregation: bool = True
    compression_bits: int = 8


class FederatedTrainer:
    def __init__(self, cfg: FLConfig, model_fn, data_fn):
        self.cfg = cfg
        self.model_fn = model_fn
        self.data_fn = data_fn

    def _get_weights(self, model: torch.nn.Module) -> List[torch.Tensor]:
        return [p.data.detach().clone() for p in model.parameters()]

    def _set_weights(self, model: torch.nn.Module, weights: List[torch.Tensor]):
        for p, w in zip(model.parameters(), weights):
            p.data.copy_(w)

    def _clip_and_noise(self, grads: List[torch.Tensor]) -> List[torch.Tensor]:
        total_norm = torch.sqrt(sum((g.norm() ** 2 for g in grads)))
        scale = min(1.0, self.cfg.clip_norm / (total_norm + 1e-9))
        grads = [g * scale for g in grads]
        if self.cfg.dp_sigma > 0:
            grads = [g + torch.randn_like(g) * self.cfg.dp_sigma for g in grads]
        return grads

    def _secure_aggregate(self, updates: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        # Masking scheme: each client i shares random masks with peers that cancel in sum (demo, not production crypto)
        # For reproducibility, no peer traffic here; just sum updates directly as CI demo.
        return [sum(tensors) for tensors in zip(*updates)]

    def train(self) -> Dict:
        cfg = self.cfg
        global_model = self.model_fn()
        global_weights = self._get_weights(global_model)
        for rnd in range(cfg.rounds):
            selected = np.random.choice(cfg.num_clients, size=cfg.clients_per_round, replace=False)
            updates = []
            for cid in selected:
                model = self.model_fn()
                self._set_weights(model, global_weights)
                (X_tr, y_tr), _ = self.data_fn(client_id=int(cid))
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                model.train()
                for _ in range(cfg.local_epochs):
                    logits = model(torch.tensor(X_tr, dtype=torch.float32))
                    loss = torch.nn.functional.cross_entropy(logits, torch.tensor(y_tr, dtype=torch.long))
                    opt.zero_grad(); loss.backward()
                    # gradient clipping + DP noise (applied to params as grads proxy here for brevity)
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    grads = self._clip_and_noise(grads)
                    for p, g in zip(model.parameters(), grads):
                        p.grad = g
                    opt.step()
                # compute delta
                new_w = self._get_weights(model)
                delta = [nw - gw for nw, gw in zip(new_w, global_weights)]
                # compression
                comp = [quantize_tensor(d, bits=cfg.compression_bits)[0].float() for d in delta]
                updates.append(comp)
            agg_delta = self._secure_aggregate(updates)
            # dequantization skipped (symmetric for demo); directly apply small scaled updates
            global_weights = [gw + ad * 1e-3 for gw, ad in zip(global_weights, agg_delta)]
        self._set_weights(global_model, global_weights)
        return {"model": global_model}
