import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv, global_mean_pool, global_max_pool
from BaselineP import load_bench

# =====================================================================
# CHUNK 1: DATA CONVERSION (bench_to_pyg)
# =====================================================================
def is_aig(G_raw) -> bool:
    allowed_gates = {"INPUT", "AND"}
    for _, node_data in G_raw.nodes(data=True):
        if node_data.get("gate", "INPUT") not in allowed_gates:
            return False
    return True

def compute_node_levels(G_raw) -> dict:
    node_levels = {}
    for level, generation in enumerate(nx.topological_generations(G_raw)):
        for node in generation: 
            node_levels[node] = level
    return node_levels

def bench_to_pyg(G_raw):
    #Converts a NetworkX graph G_raw into a PyTorch Geometric Data object.
    if not is_aig(G_raw):
        print("[REJECTED] Non-AIG graph detected.")
        return None

    node_levels = compute_node_levels(G_raw)
    nodes_list = list(G_raw.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes_list)}
    GATE_TYPES = ['INPUT', 'AND', 'NAND', 'OR', 'NOR']

    x = []
    for node in nodes_list:
        fan_in = G_raw.in_degree(node)
        fan_out = G_raw.out_degree(node)
        level = node_levels[node]
        g_type = G_raw.nodes[node].get('gate', 'INPUT')
        
        type_vec = [0.0] * 5 
        if g_type in GATE_TYPES:
            type_vec[GATE_TYPES.index(g_type)] = 1.0

        x.append(type_vec + [float(fan_in), float(fan_out), float(level)])

    edge_src, edge_dst, edge_type = [], [], []
    for u, v, edge_data in G_raw.edges(data=True):
        edge_src.append(node_to_idx[u])
        edge_dst.append(node_to_idx[v])
        edge_type.append(1 if edge_data.get("inverted", False) else 0)

    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
        edge_type=torch.tensor(edge_type, dtype=torch.long)
    )

# =====================================================================
# CHUNKS 2, 3 & 4: GNN MODEL ARCHITECTURE & POOLING
# =====================================================================
class CircuitGNN(nn.Module):
    def __init__(self, in_channels=8, hidden_channels=32, out_channels=64, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations=2))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations=2))
        self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations=2))

    def forward(self, x, edge_index, edge_type, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = x
        for conv in self.convs[:-1]:
            h = F.relu(conv(h, edge_index, edge_type))
        h_node = self.convs[-1](h, edge_index, edge_type)

        # Mean ⊕ Max Pooling (Chunk 4)
        mean_pool = global_mean_pool(h_node, batch)
        max_pool = global_max_pool(h_node, batch)
        return torch.cat([mean_pool, max_pool], dim=-1) # Returns [Batch_Size, 128]

# =====================================================================
# CHUNK 5: CONTRASTIVE LOSS (InfoNCE)
# =====================================================================
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, z_a, z_p):
        z_a = F.normalize(z_a, p=2, dim=1)
        z_p = F.normalize(z_p, p=2, dim=1)
        sim_matrix = torch.matmul(z_a, z_p.T) / self.temperature
        labels = torch.arange(z_a.size(0), device=z_a.device)
        return (self.ce(sim_matrix, labels) + self.ce(sim_matrix.T, labels)) / 2.0

# =====================================================================
# CHUNK 6: RETRIEVAL ENGINE
# =====================================================================
class CircuitRetriever:
    def __init__(self):
        self.names = []
        self.db_vectors = None

    def add_to_index(self, names, embeddings):
        norm_emb = F.normalize(embeddings, p=2, dim=-1).detach().cpu()
        if self.db_vectors is None:
            self.db_vectors = norm_emb
        else:
            self.db_vectors = torch.cat([self.db_vectors, norm_emb], dim=0)
        self.names.extend(names)

    def search(self, query_vec, top_k=3):
        if query_vec.dim() == 1:
            query_vec = query_vec.unsqueeze(0)
        q_norm = F.normalize(query_vec.detach().cpu(), p=2, dim=-1)
        sims = torch.matmul(q_norm, self.db_vectors.T).squeeze(0)
        top_scores, top_indices = torch.topk(sims, k=min(top_k, len(self.names)))
        return [(self.names[idx.item()], top_scores[i].item()) for i, idx in enumerate(top_indices)]
# ============================================================================================
# Execution of the script
# ============================================================================================
def embed_folder(model, folder_path, device="cpu"):
    model.eval()
    embeddings = {}

    for fname in os.listdir(folder_path):
        if not fname.endswith(".bench"):
            continue
        path = os.path.join(folder_path, fname)
        G_raw = load_bench(path)
        data = bench_to_pyg(G_raw)
        if data is None:  
            continue

        with torch.no_grad():
            vec = model(data.x.to(device), data.edge_index.to(device), data.edge_type.to(device))
        embeddings[fname] = vec.squeeze(0)  

    return embeddings

def compute_all_distances(embeddings, output_path):
    names = list(embeddings.keys())
    vecs = torch.stack([embeddings[n] for n in names])
    vecs_norm = F.normalize(vecs, p=2, dim=1)
    sim_matrix = torch.matmul(vecs_norm, vecs_norm.T)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["circuit_a", "circuit_b", "cosine_similarity"])
        for i, name_a in enumerate(names):
            for j, name_b in enumerate(names):
                if i < j:  
                    writer.writerow([name_a, name_b, sim_matrix[i, j].item()])

    print(f"Wrote {len(names)*(len(names)-1)//2} pairwise comparisons to {output_path}")

if __name__ == "__main__":
    G_test = nx.DiGraph()
    G_test.add_node("a", gate="INPUT")
    G_test.add_node("b", gate="INPUT")
    G_test.add_node("n1", gate="AND")
    G_test.add_edge("a", "n1", inverted=False)
    G_test.add_edge("b", "n1", inverted=True)

    data = bench_to_pyg(G_test)
    print(data.x.shape, data.edge_index.shape, data.edge_type.shape)
    model = CircuitGNN()
    embeddings = embed_folder(model, "Circuits_folder")
    compute_all_distances(embeddings, "similarity_results.csv")