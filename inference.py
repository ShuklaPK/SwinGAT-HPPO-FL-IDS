from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from utils.common import load_yaml
from data_pipeline import SDNIoTPipeline
from models.swingat_net import SwinGATNet

app = FastAPI(title="SwinGAT‑HPPO‑FL Inference")


class Flow(BaseModel):
    features: list[float]


# Load model on startup (tiny demo model)
_cfg = load_yaml("configs/experiment.yaml")
_data_cfg = load_yaml(_cfg["data_config"]) if isinstance(_cfg.get("data_config"), str) else _cfg["data_config"]
_pipe = SDNIoTPipeline(_data_cfg)
X_tr, y_tr, X_val, y_val, X_te, y_te, edge_index_np, _ = _pipe.build()
_input_dim = min(X_tr.shape[1], load_yaml(_cfg["model_config"])["input_dim"])
_model = SwinGATNet(input_dim=_input_dim)
_model.load_state_dict(torch.load("checkpoints/swingat_demo.pt", map_location="cpu"), strict=False)
_model.eval()
_edge_index = torch.tensor(edge_index_np, dtype=torch.long)


@app.post("/predict")
async def predict(flow: Flow):
    x = torch.tensor(np.array([flow.features[:_input_dim]]), dtype=torch.float32)
    with torch.no_grad():
        prob = torch.softmax(_model(x, _edge_index), -1)[0, 1].item()
    return {"malicious_prob": prob}
