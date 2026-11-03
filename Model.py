import os
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import csv
import pandas as pd
import networkx as nx
import subprocess
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.nn import RGCNConv, global_mean_pool, global_max_pool
from torch.utils.data import Dataset, DataLoader

try: # Safety mesure to ensure that program still runs
    from BaselineP import load_bench
except ImportError:
    load_bench = None

# =====================================================================
# Data Conversion (bench_to_pyg)
# =====================================================================
def is_aig(G_raw) -> bool:
    allowed_gates = {"INPUT", "AND"}
    for _, node_data in G_raw.nodes(data = True):
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
    for u, v, edge_data in G_raw.edges(data = True):
        edge_src.append(node_to_idx[u])
        edge_dst.append(node_to_idx[v])
        edge_type.append(1 if edge_data.get("inverted", False) else 0)

    return Data(
        x = torch.tensor(x, dtype = torch.float),
        edge_index = torch.tensor([edge_src, edge_dst], dtype = torch.long),
        edge_type = torch.tensor(edge_type, dtype = torch.long)
    )

# =====================================================================
# Data Preprocessing
# =====================================================================
class CircuitDataset(Dataset):
    def __init__(self, csv_path, split = "train", role = "view"):
        df = pd.read_csv(csv_path)
        df = df[(df["split"] == split) & (df["role"] == role)]
        self.function_groups = df.groupby("function_id")["circuit_path"].apply(list).to_dict()
        self.function_ids = list(self.function_groups.keys())

    def __len__(self):
        return len(self.function_ids)

    def __getitem__(self, idx):
        f_id = self.function_ids[idx]
        paths = self.function_groups[f_id]
        if len(paths) >= 2:
            p1, p2 = torch.randperm(len(paths))[:2]
            path_a, path_p = paths[p1], paths[p2]
        else: 
            path_a = path_p = paths[0]
        return path_a, path_p

# =====================================================================
# GNN Model
# =====================================================================
class CircuitGNN(nn.Module):
    def __init__(self, in_channels = 8, hidden_channels = 32, out_channels = 64, num_layers = 3):
        super().__init__()
        self.convs = nn.ModuleList()
        numrel = 2
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations = numrel))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations = numrel))
        self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations = numrel))

    def forward(self, x, edge_index, edge_type, batch = None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype = torch.long, device = x.device)
        h = x
        for conv in self.convs[:-1]:
            h = F.relu(conv(h, edge_index, edge_type))
        h_node = self.convs[-1](h, edge_index, edge_type)
        mean_pool = global_mean_pool(h_node, batch)
        max_pool = global_max_pool(h_node, batch)
        return torch.cat([mean_pool, max_pool], dim = -1) 

# =====================================================================
# Loss Factor (InfoNCE)
# =====================================================================
class InfoNCELoss(nn.Module):
    def __init__(self, temperature = 0.07):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, z_a, z_p, extra_neg = None):
        z_a = F.normalize(z_a, p = 2, dim = 1)
        z_p = F.normalize(z_p, p = 2, dim = 1)

        if extra_neg is not None:
            cols = torch.cat([z_p, F.normalize(extra_neg, p = 2, dim = 1)])
        else: 
            cols = z_p

        labels = torch.arange(z_a.size(0), device = z_a.device)
        loss_a = self.ce(z_a @ cols.T / self.temperature, labels)
        loss_p = self.ce(z_p @ z_a.T / self.temperature, labels)

        return (loss_a + loss_p) / 2.0

# =====================================================================
# Retrival Engine
# =====================================================================
class CircuitRetriever:
    def __init__(self):
        self.names = []
        self.db_vectors = None

    def add_to_index(self, names, embeddings):
        norm_emb = F.normalize(embeddings, p = 2, dim = -1).detach().cpu()
        if self.db_vectors is None:
            self.db_vectors = norm_emb
        else:
            self.db_vectors = torch.cat([self.db_vectors, norm_emb], dim = 0)
        self.names.extend(names)

    def search(self, query_vec, top_k = 3):
        if query_vec.dim() == 1:
            query_vec = query_vec.unsqueeze(0)
        q_norm = F.normalize(query_vec.detach().cpu(), p = 2, dim = -1)
        sims = torch.matmul(q_norm, self.db_vectors.T).squeeze(0)
        top_scores, top_indices = torch.topk(sims, k=min(top_k, len(self.names)))
        return [(self.names[idx.item()], top_scores[i].item()) for i, idx in enumerate(top_indices)]
        
# ============================================================================================
# Circuit Equivalence Checking
# ============================================================================================
def run_abc_cec(file1, file2) -> bool:
    try:
        cmd = f"read {file1}; read {file2}; cec"
        result = subprocess.run(
            ["abc", "-c", cmd],
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
            timeout = 10
        )
        output = result.stdout + result.stderr
        if "Networks are equivalent" in output:
            return True
        return False
    except FileNotFoundError:
        print("[Warning] 'abc' binary not found in system PATH. Failing closed (returning False).")
        return False  # <-- CHANGED FROM True TO False
    except Exception as e:
        print(f"[Error] ABC CEC failed: {e}")
        return False

def cec_checking(query_path, candidates, folder_path="Circuits_folder"):
    verified_results = []
    for name, score in candidates:
        cand_path = os.path.join(folder_path, name)
        if run_abc_cec(query_path, cand_path):
            verified_results.append((name, score))
    return verified_results

def convert_batch(path_list, device):
    data_list = []
    for path in path_list:
        G_raw = load_bench(path)
        if G_raw is not None:  # <-- Added check
            data = bench_to_pyg(G_raw)
            if data is not None:
                data_list.append(data)
    if not data_list:      # <-- Added check
        return None
    return Batch.from_data_list(data_list).to(device)

# ============================================================================================
# Execution of the script
# ============================================================================================
def train_circuit_model(model, csv_path, epochs = 10, batch_size = 32, device = "cpu"):
    dataset = CircuitDataset(csv_path, split = "train", role = "view")
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    criterion = InfoNCELoss(temperature = 0.07)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        valid_batches = 0  # <-- Track valid batches
        for path_a_list, path_p_list in dataloader:
            batch_a = convert_batch(path_a_list, device)
            batch_p = convert_batch(path_p_list, device)
            
            if batch_a is None or batch_p is None:  # <-- Skip bad batches
                continue
            
            optimizer.zero_grad()
            z_a = model(batch_a.x, batch_a.edge_index, batch_a.edge_type, batch_a.batch)
            z_p = model(batch_p.x, batch_p.edge_index, batch_p.edge_type, batch_p.batch)
            
            loss = criterion(z_a, z_p)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            valid_batches += 1
            
        if valid_batches == 0:
            print(f"[Warning] Epoch {epoch+1}/{epochs}: 0 valid batches processed! Check if samples.csv paths point to valid AIG (.bench) files.")
        else:
            avg_loss = total_loss / valid_batches
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

# ============================================================================================
# Execution of the script
# ============================================================================================
def embed_folder(model, folder_path, device = "cpu"):
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
    vecs_norm = F.normalize(vecs, p = 2, dim = 1)
    sim_matrix = torch.matmul(vecs_norm, vecs_norm.T)

    with open(output_path, "w", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(["circuit_a", "circuit_b", "cosine_similarity"])
        for i, name_a in enumerate(names):
            for j, name_b in enumerate(names):
                if i < j:  
                    writer.writerow([name_a, name_b, sim_matrix[i, j].item()])

    print(f"Wrote {len(names)*(len(names)-1)//2} pairwise comparisons to {output_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the GNN Model
    model = CircuitGNN(in_channels = 8, hidden_channels = 32, out_channels = 64, num_layers = 3).to(device)
    
    # Training Metrics
    print("\n--- Phase 1: Training on Synthetic Circuits ---")
    start_time = time.time()
    train_circuit_model(model, csv_path = "samples.csv", epochs = 5, batch_size = 16, device = device)
    train_duration = time.time() - start_time
    print(f"[Metrics] Training completed in {train_duration:.2f} seconds.")
    
    # --- Phase 2: Embedding Folders ---
    print("\n--- Phase 2: Embedding Folders ---")
    embeddings = embed_folder(model, "Circuits_folder", device = device)
    
    if embeddings:
        #To prevent OOM / crashes with >800 circuits, compute pairwise distances in chunks if needed, 
        # or skip full N*N matrix generation if only searching. Here we safely use the retriever index:
        retriever = CircuitRetriever()
        names = list(embeddings.keys())
        vecs = torch.stack([embeddings[n] for n in names])
        retriever.add_to_index(names, vecs)
        
        # --- Phase 3: Testing with a Target Circuit & Top 20 CEC ---
        print("\n--- Phase 3: Testing Target Circuit & Top 20 CEC Verification ---")
        # Select target circuit (e.g., from your list of 8 test circuits or first available)
        target_circuit = names[0] if names else None
        if target_circuit:
            query_path = os.path.join("Circuits_folder", target_circuit)
            query_vec = embeddings[target_circuit]
            
            # Get top 20 retrieved candidates
            top_candidates = retriever.search(query_vec, top_k = 20)
            print(f"Top-20 raw matches for query '{target_circuit}':")
            for rank, (cand_name, score) in enumerate(top_candidates, 1):
                print(f"  {rank}. {cand_name} (Cosine Similarity: {score: .4f})")
            
            # Run CEC verification on top 20 candidates + display scores
            print("\nRunning Berkeley ABC CEC on Top-20 Candidates...")
            verified_results = cec_checking(query_path, top_candidates, folder_path = "Circuits_folder")
            print(f"\n[Results] CEC Verified Equivalent Matches out of Top 20:")
            for name, score in verified_results:
                print(f"  - {name} | Similarity: {score: .4f} | Status: Equivalent")