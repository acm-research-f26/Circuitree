import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, z_a: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:

        z_a = F.normalize(z_a, p=2, dim=1)
        z_p = F.normalize(z_p, p=2, dim=1)

        sim_matrix = torch.matmul(z_a, z_p.T) / self.temperature

        batch_size = z_a.size(0)
        labels = torch.arange(batch_size, device=z_a.device)

        loss_a_to_p = self.cross_entropy(sim_matrix, labels)
        loss_p_to_a = self.cross_entropy(sim_matrix.T, labels)

        return (loss_a_to_p + loss_p_to_a) / 2.0