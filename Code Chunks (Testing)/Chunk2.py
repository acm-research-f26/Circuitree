import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class CircuitRGCNLayer(nn.Module):

    def __init__(self, in_channels: int = 8, out_channels: int = 32, num_relations: int = 2):
        super().__init__()
        self.conv = RGCNConv(
            in_channels=in_channels,
            out_channels=out_channels,
            num_relations=num_relations,
            num_bases=None  # Full relation-specific weight matrices W_0 and W_1
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index, edge_type)
        return F.relu(h)