from utils.common import load_yaml
from data_pipeline import SDNIoTPipeline


def test_pipeline_shapes():
    cfg = load_yaml("configs/data_sdniot.yaml")
    X_tr, y_tr, X_val, y_val, X_te, y_te, edge_index, edge_weight = SDNIoTPipeline(cfg).build()
    assert X_tr.shape[0] > 0 and edge_index.shape[0] == 2
