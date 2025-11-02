import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np

from utils.seed import set_deterministic
from utils.common import load_yaml, ensure_dir
from utils.metrics import compute_basic_metrics
from data_pipeline import SDNIoTPipeline
from models.swingat_net import SwinGATNet
from feature_selection.caps_gjo import CapsGJOSelector
from federated.fl_trainer import FederatedTrainer, FLConfig


def get_data(cfg_data):
    pipe = SDNIoTPipeline(cfg_data)
    return pipe.build()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/experiment.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--use_fl", action="store_true")
    args = ap.parse_args()

    top = load_yaml(args.config)
    set_deterministic(top.get("seed", 42))
    ensure_dir(top.get("checkpoint_dir", "./checkpoints"))

    data_cfg = load_yaml(top["data_config"]) if isinstance(top.get("data_config"), str) else top["data_config"]
    X_tr, y_tr, X_val, y_val, X_te, y_te, edge_index_np, edge_weight = get_data(data_cfg)

    # Optional Caps-GJO
    if top.get("features", "all") == "caps_gjo":
        caps_cfg = load_yaml(top["caps_gjo_config"]) if isinstance(top.get("caps_gjo_config"), str) else top["caps_gjo_config"]
        selector = CapsGJOSelector(caps_cfg)
        sel_idx = selector.select_k(X_tr.toarray() if hasattr(X_tr, "toarray") else X_tr, y_tr)
    else:
        sel_idx = list(range(min(X_tr.shape[1], load_yaml(top["model_config"])["input_dim"])))

    def to_tensor(X):
        Xn = (X.toarray() if hasattr(X, "toarray") else np.asarray(X))[:, sel_idx]
        return torch.tensor(Xn, dtype=torch.float32)

    Xtr_t, Xval_t, Xte_t = map(to_tensor, [X_tr, X_val, X_te])
    ytr_t, yval_t, yte_t = map(lambda a: torch.tensor(a, dtype=torch.long), [y_tr, y_val, y_te])

    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    model_cfg = load_yaml(top["model_config"]) if isinstance(top.get("model_config"), str) else top["model_config"]
    model = SwinGATNet(
        input_dim=len(sel_idx),
        num_classes=model_cfg.get("num_classes", 2),
        swin_name=model_cfg["swin"]["model_name"],
        img_adapter=model_cfg["swin"]["img_adapter"],
        img_size=model_cfg["swin"]["img_size"],
        swin_drop=model_cfg["swin"].get("drop_rate", 0.1),
        gat_hidden=model_cfg["gat"]["hidden_dim"],
        gat_heads=model_cfg["gat"]["num_heads"],
        gat_layers=model_cfg["gat"]["num_layers"],
        gat_dropout=model_cfg["gat"]["dropout"],
        fusion_dim=model_cfg.get("fusion_dim", 128),
        label_smoothing=top.get("label_smoothing", 0.0),
        use_focal=top.get("use_focal", False),
    )

    if args.use_fl or top.get("use_fl", False):
        # Federated demo
        def model_fn():
            return SwinGATNet(input_dim=len(sel_idx))

        def data_fn(client_id: int):
            # partition data by client id
            n = Xtr_t.shape[0]
            idx = np.arange(n)
            parts = np.array_split(idx, load_yaml(top["fl_config"])["num_clients"])
            ids = parts[client_id]
            return (Xtr_t[ids].numpy(), ytr_t[ids].numpy()), None

        fl_cfg = FLConfig(**load_yaml(top["fl_config"]))
        trainer = FederatedTrainer(fl_cfg, model_fn=model_fn, data_fn=data_fn)
        res = trainer.train()
        model = res["model"]
    else:
        # Central training
        opt = torch.optim.Adam(model.parameters(), lr=top.get("lr", 1e-3), weight_decay=top.get("weight_decay", 1e-4))
        epochs = args.epochs or top.get("epochs", 3)
        for ep in range(epochs):
            model.train()
            logits = model(Xtr_t, edge_index)
            loss = F.cross_entropy(logits, ytr_t, label_smoothing=top.get("label_smoothing", 0.0))
            opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                prob = torch.softmax(model(Xval_t, edge_index), -1)[:, 1].numpy()
                m = compute_basic_metrics(y_val, prob)
            print(f"Epoch {ep+1}/{epochs} loss={loss.item():.4f} val_acc={m['accuracy']:.3f} fpr={m['fpr']:.3f}")

    # Save checkpoint
    ensure_dir("checkpoints")
    torch.save(model.state_dict(), "checkpoints/swingat_demo.pt")

    # Quick test metrics
    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(Xte_t, edge_index), -1)[:, 1].numpy()
        m = compute_basic_metrics(y_te, prob)
    print("Test:", m)


if __name__ == "__main__":
    main()
