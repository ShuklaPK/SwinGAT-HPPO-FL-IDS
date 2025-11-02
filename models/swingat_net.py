import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


class SimpleGATLayer(nn.Module):
    """Minimal GAT layer in PyTorch using edge_index (2,E).
    Supports multi-head attention and dropout.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.h = num_heads
        self.out_dim = out_dim
        self.lin = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(num_heads, out_dim))
        self.att_dst = nn.Parameter(torch.Tensor(num_heads, out_dim))
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        h = self.h
        x_proj = self.lin(x).view(N, h, self.out_dim)  # (N, H, D)
        src, dst = edge_index
        x_i = x_proj[dst]  # messages to dst
        x_j = x_proj[src]
        e = (x_j * self.att_src).sum(-1) + (x_i * self.att_dst).sum(-1)  # (E, H)
        e = self.leakyrelu(e)
        # softmax over incoming edges per node and head
        # gather by dst indices
        alpha = torch.zeros_like(e)
        for head in range(h):
            alpha[:, head] = torch.zeros(N, device=x.device).index_add_(
                0, dst, torch.exp(e[:, head])
            )[dst]
        alpha = torch.exp(e) / (alpha + 1e-9)
        alpha = self.dropout(alpha)
        out = torch.zeros(N, h, self.out_dim, device=x.device)
        for head in range(h):
            out[:, head, :] = out[:, head, :].index_add_(
                0, dst, x_j[:, head, :] * alpha[:, head : head + 1]
            )
        return out.reshape(N, h * self.out_dim)


class SwinAdapter(nn.Module):
    """Maps tabular vectors to pseudo-images for Swin, then pools back to vector."""

    def __init__(self, in_features: int, img_size: int = 224, channels: int = 4):
        super().__init__()
        self.img_size = img_size
        self.channels = channels
        # project to channels * (img_size//16)^2 tokens then reshape to image-like
        tokens = (img_size // 16) ** 2 * channels
        self.proj = nn.Linear(in_features, tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        y = self.proj(x)
        y = y.view(B, self.channels, self.img_size // 16, self.img_size // 16)
        y = F.interpolate(y, size=(self.img_size, self.img_size), mode="bilinear")
        return y


class SwinGATNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        swin_name: str = "swin_tiny_patch4_window7_224",
        img_adapter: int = 4,
        img_size: int = 224,
        swin_drop: float = 0.1,
        gat_hidden: int = 64,
        gat_heads: int = 4,
        gat_layers: int = 2,
        gat_dropout: float = 0.2,
        fusion_dim: int = 128,
        label_smoothing: float = 0.0,
        use_focal: bool = False,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.use_focal = use_focal

        # Tabular -> pseudo image -> Swin backbone
        self.adapter = SwinAdapter(input_dim, img_size, img_adapter)
        self.swin = create_model(swin_name, pretrained=False, num_classes=0, drop_rate=swin_drop)
        swin_out = self.swin.num_features

        # Graph stream: simple MLP to embed tabular, then GAT layers
        self.node_embed = nn.Sequential(
            nn.Linear(input_dim, gat_hidden), nn.ReLU(), nn.Dropout(gat_dropout)
        )
        gat_dims = [gat_hidden] + [gat_hidden] * (gat_layers - 1)
        self.gats = nn.ModuleList(
            [
                SimpleGATLayer(d, gat_hidden // gat_heads, num_heads=gat_heads, dropout=gat_dropout)
                for d in gat_dims
            ]
        )
        self.gat_post = nn.Sequential(nn.ReLU(), nn.Dropout(gat_dropout))

        # Fusion
        self.fuse = nn.Sequential(
            nn.Linear(swin_out + gat_hidden, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, num_classes),
        )

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.use_focal:
            # focal loss for binary classification
            probs = torch.sigmoid(logits.squeeze(-1) if logits.size(-1) == 1 else logits[:, 1])
            targets_f = targets.float()
            alpha, gamma = 0.25, 2.0
            pt = torch.where(targets_f == 1, probs, 1 - probs)
            loss = -alpha * (1 - pt) ** gamma * torch.log(pt + 1e-9)
            return loss.mean()
        else:
            return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)

    def forward_tabular(self, x: torch.Tensor) -> torch.Tensor:
        img = self.adapter(x)
        swin_feat = self.swin(img)
        return swin_feat

    def forward_graph(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.node_embed(x)
        for gat in self.gats:
            h = gat(h, edge_index)
        # concat heads already inside SimpleGATLayer; project back to hidden
        h = self.gat_post(h)
        return h

    def forward_mixed(
        self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        swin_feat = self.forward_tabular(x)
        gat_feat = self.forward_graph(x, edge_index) if edge_index is not None else torch.zeros(
            x.size(0), self.gats[-1].h * self.gats[-1].out_dim, device=x.device
        )
        fused = torch.cat([swin_feat, gat_feat], dim=-1)
        logits = self.fuse(fused)
        return logits, swin_feat, gat_feat

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits, _, _ = self.forward_mixed(x, edge_index)
        return logits
