from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

_NORM_MAP = {
    "batch": nn.BatchNorm1d,
    "layer": nn.LayerNorm,
    "none": None,
}
class GCN(nn.module):
    f"""
    A configurable-depth GCN with optional residual connections and per-layer normalization. Final layer outputs logits (no act/norm/dropout).

    Args:
        in_dim (int): num of input features
        hidden (int): hidden width for layers 1..L-1 (ignore if num_layers == 1)
        out_dim (int): num of classes
        dropout (float): dropout prob applied on hidden layers
        num_layers (int): total layers (1..10)
        residual (bool): add residual connections between hidden layers
        norm (str): one of {"batch", "layer", "none"}
    """
    def __init__(self, 
                in_dim: int,
                hidden: int,
                out_dim: int,
                dropout: float = 0.5,
                num_layers: int = 2,
                residual: bool = True,
                norm: str = "batch"):
        super().__init__()
        assert 1 <= num_layers <= 10, "Number of layers must be in [1, 10]"
        self.num_layers = num_layers
        self.residual = residual
        self.dropout_p = float(dropout)

        if num_layers == 1:
            # single layer gcn -> directly to logits
            self.convs = nn.ModuleList([GCNConv(in_dim, out_dim, cached=True)])
            self.norms = nn.ModuleList()
        else:
            # 1st layer
            convs = [GCNConv(in_dim, hidden, cached=True)]
            # middle layers
            for _ in range (num_layers - 2):
                convs.append(GCNConv(hidden, hidden, cached= True))
            # final layer to logits
            convs.append(GCNConv(hidden, out_dim, cached=True))
            self.convs = nn.ModuleList(convs)

            Norm = _NORM_MAP.get(norm, None)
            if Norm is None:
                self.norms = nn.ModuleList([nn.Identity() for _ in range(num_layers -1)])
            else:
                self.norms = nn.ModuleList([Norm(hidden) for _ in range(num_layers - 1)])
        
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x, edge_index):
        if self.num_layers == 1:
            return self.convs[0](x, edge_index)
        
        # hidden blocks (1..L-1)
        for i in range(self.num_layers - 1):
            h = self.convs[i](x, edge_index)
            h = self.norms[i](h)
            h = self.act(h)
            h = self.dropout(h)
            if self.residual and h.shape == x.shape:
                x = x+h
            else:
                x = h
            
        # fianl logits
        x = self.convs[-1](x, edge_index)
        return x
    
def make_gcn(num_features: int,
            num_classes: int,
            hidden: int = 64,
            dropout: float = 0.5,
            num_layers: int = 2,
            residual: bool = True,
            norm: str = "batch") -> nn.Module:
    return GCN(
        in_dim = num_features,
        hidden = hidden,
        out_dim = num_classes,
        dropout = dropout,
        num_layers = num_layers,
        residual = residual,
        norm = norm,
    )
    
    