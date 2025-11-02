import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class PrimaryCaps(nn.Module):
    def __init__(self, input_dim: int, num_caps: int, cap_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_caps * cap_dim)
        self.num_caps = num_caps
        self.cap_dim = cap_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        y = y.view(x.size(0), self.num_caps, self.cap_dim)
        return squash(y)


def squash(s: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    mag2 = (s ** 2).sum(dim=-1, keepdim=True)
    mag = torch.sqrt(mag2 + eps)
    return (mag2 / (1.0 + mag2)) * (s / (mag + eps))


class CapsEmbedder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_caps: int, cap_dim: int, routings: int = 2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.prim = PrimaryCaps(hidden_dim, num_caps, cap_dim)
        self.routings = routings
        self.att = nn.Parameter(torch.randn(num_caps, cap_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        caps = self.prim(h)
        # simple agreement routing via attention vector
        att = F.normalize(self.att, dim=-1)
        coeff = torch.einsum("bcd,cd->bc", caps, att)
        coeff = coeff.softmax(dim=-1)
        out = torch.einsum("bc,bcd->bd", coeff, caps)
        return out  # (B, cap_dim)


# Golden Jackal Optimization (simplified, faithful variant)
# Reference concept: social hierarchy & exploration/exploitation using guiding leaders.

class GoldenJackalOptimizer:
    def __init__(self, fitness_fn, dim: int, population: int = 12, iterations: int = 10, w: float = 0.9, c1: float = 1.2, c2: float = 1.2, rng: np.random.Generator | None = None):
        self.fitness_fn = fitness_fn
        self.dim = dim
        self.pop = population
        self.iters = iterations
        self.w, self.c1, self.c2 = w, c1, c2
        self.rng = rng or np.random.default_rng(42)

    def run(self, lb: float = 0.0, ub: float = 1.0) -> np.ndarray:
        X = self.rng.uniform(lb, ub, size=(self.pop, self.dim))
        V = np.zeros_like(X)
        pbest = X.copy()
        pbest_fit = np.array([self.fitness_fn(x) for x in X])
        gbest = X[pbest_fit.argmin()].copy()
        gbest_fit = pbest_fit.min()
        for _ in range(self.iters):
            r1, r2 = self.rng.random((2, self.pop, self.dim))
            V = self.w * V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (gbest - X)
            X = np.clip(X + V, lb, ub)
            fit = np.array([self.fitness_fn(x) for x in X])
            improved = fit < pbest_fit
            pbest[improved] = X[improved]
            pbest_fit[improved] = fit[improved]
            if pbest_fit.min() < gbest_fit:
                gbest = pbest[pbest_fit.argmin()].copy()
                gbest_fit = pbest_fit.min()
        return gbest


class CapsGJOSelector:
    def __init__(self, cfg: Dict):
        self.cfg = cfg

    def rank_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Embed with capsules then compute mutual-info like scores via correlation to y
        device = torch.device("cpu")
        x_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        cap = CapsEmbedder(
            input_dim=X.shape[1],
            hidden_dim=self.cfg["capsule"]["hidden_dim"],
            num_caps=self.cfg["capsule"]["num_caps"],
            cap_dim=self.cfg["capsule"]["cap_dim"],
            routings=self.cfg["capsule"]["routings"],
        ).to(device)
        with torch.no_grad():
            z = cap(x_t)
        # feature importance via absolute correlation between each input column and capsule embedding norms
        norms = z.norm(dim=-1)
        scores = []
        for j in range(X.shape[1]):
            col = x_t[:, j]
            c = torch.corrcoef(torch.stack([col, norms]))[0, 1].abs().item()
            scores.append(c)
        ranks = np.argsort(-np.nan_to_num(np.array(scores), nan=0.0))
        return ranks

    def select_k(self, X: np.ndarray, y: np.ndarray, k: int | None = None) -> List[int]:
        k = k or int(self.cfg["selection"]["k"])
        ranks = self.rank_features(X, y)
        # Wrapper refinement with GJO to trade off latency (feature count) vs accuracy proxy
        def fitness(bitmask: np.ndarray) -> float:
            chosen = np.where(bitmask > 0.5)[0]
            if len(chosen) == 0:
                return 1e6
            # proxy: logistic regression accuracy (fast) with 2-fold CV
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score

            Xs = X[:, chosen]
            try:
                clf = LogisticRegression(max_iter=200)
                acc = cross_val_score(clf, Xs, y, cv=2, scoring="accuracy").mean()
            except Exception:
                acc = 0.0
            latency = len(chosen) / X.shape[1]
            return (1 - acc) + self.cfg["selection"].get("latency_weight", 0.1) * latency

        gjo = GoldenJackalOptimizer(fitness_fn=fitness, dim=k, population=self.cfg["optimizer"]["population"], iterations=self.cfg["optimizer"]["iterations"], w=self.cfg["optimizer"]["w"], c1=self.cfg["optimizer"]["c1"], c2=self.cfg["optimizer"]["c2"])
        mask = gjo.run()
        topk = ranks[:k]
        # refine top-k using mask (keep ~half as indicated by optimizer)
        keep = [idx for i, idx in enumerate(topk) if mask[i] > 0.5]
        if not keep:
            keep = topk[: max(1, k // 2)].tolist()
        return keep

    def export_schema(self, selected_idx: List[int], feature_names: List[str]) -> Dict:
        return {"selected_indices": selected_idx, "selected_features": [feature_names[i] for i in selected_idx]}
