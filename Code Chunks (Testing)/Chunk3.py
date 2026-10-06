import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class CircuitRGCN(nn.Module):
    def __init__(
        self, 
        in_channels: int = 8, 
        hidden_channels: int = 32, 
        out_channels: int = 64, 
        num_relations: int = 2,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()

        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations=num_relations))

        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations))

        self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations=num_relations))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:

        h = x
        for i in range(self.num_layers - 1):
            h = self.convs[i](h, edge_index, edge_type)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.convs[-1](h, edge_index, edge_type)
        return h