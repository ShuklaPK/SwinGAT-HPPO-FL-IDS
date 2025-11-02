import torch
from models.swingat_net import SwinGATNet


def test_model_forward_shapes():
    B, D = 8, 32
    x = torch.randn(B, D)
    edge_index = torch.randint(0, B, (2, B * 2))
    model = SwinGATNet(input_dim=D)
    out = model(x, edge_index)
    assert out.shape == (B, 2)
