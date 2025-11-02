import argparse
import os
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump, load

from utils.seed import set_deterministic
from utils.graph_utils import build_adjacency_knn


class SDNIoTPipeline:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.cache_dir = Path(cfg.get("cache_dir", "./.cache/sdniot"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_csvs(self, data_path: Path) -> pd.DataFrame:
        if not data_path.exists():
            # Synthetic fallback: simple numeric/categorical mix resembling flows
            n = 200
            rng = np.random.default_rng(42)
            df = pd.DataFrame(
                {
                    "src": rng.integers(0, 50, n),
                    "dst": rng.integers(0, 50, n),
                    "bytes": rng.lognormal(5, 1, n),
                    "packets": rng.integers(1, 100, n),
                    "duration": rng.random(n) * 5,
                    "proto": rng.choice(["tcp", "udp", "icmp"], n),
                    "service": rng.choice(["http", "dns", "ssh", "none"], n),
                    "state": rng.choice(["EST", "INT", "FIN"], n),
                    "ts": rng.integers(1_700_000_000, 1_700_000_000 + 3600, n),
                }
            )
            # label: simple rule + noise
            df["label"] = ((df["bytes"] > 200) & (df["service"] == "none")).astype(int)
            return df
        # Load all CSVs
        frames = []
        for p in data_path.glob("*.csv"):
            frames.append(pd.read_csv(p))
        if not frames:
            raise FileNotFoundError(f"No CSVs found in {data_path}")
        return pd.concat(frames, ignore_index=True)

    def build(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.cfg
        data_path = Path(cfg.get("data_path", "./datasets/sdniot"))
        label_col = cfg.get("label", "label")
        categoricals = cfg.get("categoricals", [])
        val_ratio = float(cfg.get("val_ratio", 0.1))
        test_ratio = float(cfg.get("test_ratio", 0.1))
        standardize = bool(cfg.get("standardize", True))
        knn_k = int(cfg.get("knn_k", 8))

        df = self._load_csvs(Path(data_path))
        df = df.dropna().reset_index(drop=True)
        y = df[label_col].astype(int).values
        X_df = df.drop(columns=[label_col])

        # Separate numeric and categorical columns
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in categoricals if c in X_df.columns]

        transformers = []
        if cat_cols:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
        if numeric_cols:
            if standardize:
                transformers.append(("num", StandardScaler(), numeric_cols))
            else:
                transformers.append(("num", "passthrough", numeric_cols))

        pre = ColumnTransformer(transformers)
        pipe = Pipeline([("pre", pre)])
        X = pipe.fit_transform(X_df)

        # Cache preprocessor
        dump(pipe, self.cache_dir / "preprocessor.joblib")

        # Train/val/test split
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=val_ratio + test_ratio, random_state=42, stratify=y)
        rel = test_ratio / (val_ratio + test_ratio)
        X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=rel, random_state=42, stratify=y_tmp)

        # kNN graph on train embeddings to define topology; reuse for val/test via nearest neighbors to train set
        X_tr_arr = X_tr.toarray() if hasattr(X_tr, "toarray") else np.asarray(X_tr)
        edge_index, edge_weight = build_adjacency_knn(X_tr_arr, k=knn_k)
        dump({"edge_index": edge_index, "edge_weight": edge_weight}, self.cache_dir / "graph.joblib")

        return X_tr, y_tr, X_val, y_val, X_te, y_te, edge_index, edge_weight

    def load_cached(self):
        pre = load(self.cache_dir / "preprocessor.joblib")
        graph = load(self.cache_dir / "graph.joblib")
        return pre, graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/data_sdniot.yaml")
    ap.add_argument("--build_cache", action="store_true")
    args = ap.parse_args()

    from utils.common import load_yaml

    cfg = load_yaml(args.config)
    set_deterministic(42)
    pipe = SDNIoTPipeline(cfg)
    X_tr, y_tr, X_val, y_val, X_te, y_te, edge_index, edge_weight = pipe.build()
    print("Shapes:", [
        getattr(X_tr, "shape", None), getattr(X_val, "shape", None), getattr(X_te, "shape", None),
        edge_index.shape, edge_weight.shape
    ])
    if args.build_cache:
        print("Cache built at", pipe.cache_dir)


if __name__ == "__main__":
    main()
