import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from typing import Tuple


def build_adjacency_knn(X: np.ndarray, k: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Return edge_index (2,E) and edge_weight (E,) using kNN over embeddings.
    Self-loops included.
    """
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=min(k, n)).fit(X)
    dists, idxs = nbrs.kneighbors(X)
    sources = np.repeat(np.arange(n), idxs.shape[1])
    targets = idxs.reshape(-1)
    weights = np.exp(-dists.reshape(-1))
    # add self-loops
    sources = np.concatenate([sources, np.arange(n)])
    targets = np.concatenate([targets, np.arange(n)])
    weights = np.concatenate([weights, np.ones(n)])
    return np.vstack([sources, targets]).astype(np.int64), weights.astype(np.float32)


def graph_stats(edge_index: np.ndarray, n: int) -> dict:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    G.add_edges_from(edges)
    return {
        "nodes": n,
        "edges": int(edge_index.shape[1]),
        "components": nx.number_connected_components(G),
        "avg_degree": 2 * G.number_of_edges() / max(1, G.number_of_nodes()),
    }
