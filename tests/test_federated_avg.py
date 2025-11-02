import numpy as np
from federated.fl_trainer import FederatedTrainer, FLConfig
from models.swingat_net import SwinGATNet


def test_federated_run():
    def model_fn():
        return SwinGATNet(input_dim=16)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(96, 16)).astype("float32")
    y = (X[:, 0] > 0).astype("int64")

    def data_fn(client_id: int):
        parts = np.array_split(np.arange(96), 3)
        ids = parts[client_id]
        return (X[ids], y[ids]), None

    fl = FederatedTrainer(FLConfig(rounds=1, local_epochs=1), model_fn, data_fn)
    res = fl.train()
    assert res["model"] is not None
