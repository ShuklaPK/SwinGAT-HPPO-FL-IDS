import argparse
import torch
import numpy as np
from utils.metrics import compute_basic_metrics, latency_percentiles
from data_pipeline import SDNIoTPipeline
from models.swingat_net import SwinGATNet
from utils.common import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/swingat_demo.pt")
    ap.add_argument("--config", type=str, default="configs/experiment.yaml")
    args = ap.parse_args()

    top = load_yaml(args.config)
    data_cfg = load_yaml(top["data_config"]) if isinstance(top.get("data_config"), str) else top["data_config"]
    X_tr, y_tr, X_val, y_val, X_te, y_te, edge_index_np, _ = SDNIoTPipeline(data_cfg).build()

    input_dim = min(X_tr.shape[1], load_yaml(top["model_config"])["input_dim"])
    model = SwinGATNet(input_dim=input_dim)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    def to_tensor(X):
        return torch.tensor((X.toarray() if hasattr(X, "toarray") else X)[:, :input_dim], dtype=torch.float32)

    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    with torch.no_grad():
        prob = torch.softmax(model(to_tensor(X_te), edge_index), -1)[:, 1].numpy()
        m = compute_basic_metrics(y_te, prob)
    print("Metrics:", m)

    # Fake latency profiling around model forward
    import time
    lat = []
    x = to_tensor(X_te[:32])
    for _ in range(16):
        t0 = time.perf_counter(); _ = model(x, edge_index[: , : min(256, edge_index.shape[1])]); lat.append((time.perf_counter()-t0)*1000)
    print("Latency:", latency_percentiles(lat))


if __name__ == "__main__":
    main()
