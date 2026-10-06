import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict

class CircuitRetriever:
    def __init__(self):
        self.circuit_names: List[str] = []
        self.db_embeddings: torch.Tensor = None  

    def add_circuits(self, names: List[str], embeddings: torch.Tensor):
        norm_embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        if self.db_embeddings is None:
            self.db_embeddings = norm_embeddings.detach().cpu()
        else:
            self.db_embeddings = torch.cat([self.db_embeddings, norm_embeddings.detach().cpu()], dim=0)
            
        self.circuit_names.extend(names)

    def search(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.db_embeddings is None or len(self.circuit_names) == 0:
            raise ValueError("Database is empty. Add circuit embeddings before querying.")

        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
            
        q_norm = F.normalize(query_embedding.detach().cpu(), p=2, dim=-1)

        similarities = torch.matmul(q_norm, self.db_embeddings.T).squeeze(0)

        actual_k = min(top_k, len(self.circuit_names))
        top_scores, top_indices = torch.topk(similarities, k=actual_k, largest=True)

        results = [
            (self.circuit_names[idx.item()], top_scores[i].item())
            for i, idx in enumerate(top_indices)
        ]
        return results

    def evaluate_recall_at_k(self, query_embeddings: torch.Tensor, ground_truth_names: List[str], k: int = 1) -> float:

        correct = 0
        total = len(ground_truth_names)

        for i in range(total):
            results = self.search(query_embeddings[i], top_k=k)
            retrieved_names = [name for name, _ in results]
            if ground_truth_names[i] in retrieved_names:
                correct += 1

        return correct / total if total > 0 else 0.0