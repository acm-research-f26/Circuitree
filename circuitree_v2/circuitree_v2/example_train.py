"""Minimal end-to-end example: train a relational GNN with a contrastive loss, then evaluate.

This only shows the data flowing through a model; it is not tuned and not a reference result.

    pip install torch torch_geometric numpy
    python example_train.py --track synthetic --steps 500

Model: node features = one-hot type + input position; 3 RGCN layers with four edge relations
(plain/inverted x forward/backward); mean+max pooling; projection to a unit vector.
Loss: InfoNCE. Each batch holds two views of B different functions; the other function's views
(and, with --near-miss, one-gate mutants of the batch functions) are the negatives.
"""
import argparse
import random
import os
import subprocess
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import RGCNConv, global_max_pool, global_mean_pool

import circuitree_data as cd
import evaluate as ev


class CircuitEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden=128, out=128, layers=3):
        super().__init__()
        self.inp = torch.nn.Linear(in_dim, hidden)
        self.convs = torch.nn.ModuleList(RGCNConv(hidden, hidden, num_relations=4) for _ in range(layers))
        self.head = torch.nn.Sequential(torch.nn.Linear(2 * hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, out))

    def forward(self, data):
        # Add reverse edges so information also flows from outputs back to inputs.
        ei = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
        et = torch.cat([data.edge_type, data.edge_type + 2])
        h = F.relu(self.inp(data.x))
        for conv in self.convs:
            h = F.relu(conv(h, ei, et)) + h
        g = torch.cat([global_mean_pool(h, data.batch), global_max_pool(h, data.batch)], dim=1)
        return F.normalize(self.head(g), dim=1)


def info_nce(za, zb, extra_neg=None, temperature=0.1):
    """za[i] and zb[i] are two views of function i; everything else is a negative."""
    logits = za @ zb.t()
    if extra_neg is not None and len(extra_neg):
        logits = torch.cat([logits, za @ extra_neg.t()], dim=1)
    target = torch.arange(len(za))
    return F.cross_entropy(logits / temperature, target)


def embed(model, graphs, batch_size=256):
    model.eval()
    out = {}
    with torch.no_grad():
        for i in range(0, len(graphs), batch_size):
            chunk = graphs[i:i + batch_size]
            z = model(Batch.from_data_list(chunk))
            for d, v in zip(chunk, z):
                out[d.sample_id] = v.numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", default = "synthetic", choices = ["real", "synthetic"])
    p.add_argument("--steps", type = int, default = 1000)
    p.add_argument("--batch-functions", type = int, default = 64)
    p.add_argument("--near-miss", action = "store_true", help = "add one-gate mutants as extra negatives")
    p.add_argument("--no-positions", action = "store_true", help = "drop the input-position feature")
    p.add_argument("--seed", type = int, default = 0)
    args = p.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    positional = not args.no_positions
    train = cd.load_pyg(args.track, "train", positional = positional)
    by_fn, mutants_of = {}, {}
    rows = {r["sample_id"]: r for r in cd.read_samples(track = args.track, split="train")}
    for d in train:
        r = rows[d.sample_id]
        if r["role"] == "view":
            by_fn.setdefault(d.function_id, []).append(d)
        else:
            mutants_of.setdefault(r["source_function_id"], []).append(d)
    functions = [f for f, vs in by_fn.items() if len(vs) >= 2]
    print(f"train: {len(train)} graphs, {len(functions)} functions with >= 2 views")

    model = CircuitEncoder(train[0].x.size(1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(1, args.steps + 1):
        model.train()
        batch_fns = random.sample(functions, min(args.batch_functions, len(functions)))
        pairs = [random.sample(by_fn[f], 2) for f in batch_fns]
        za = model(Batch.from_data_list([a for a, _ in pairs]))
        zb = model(Batch.from_data_list([b for _, b in pairs]))
        extra = None
        if args.near_miss:
            negs = [random.choice(mutants_of[f]) for f in batch_fns if f in mutants_of]
            extra = model(Batch.from_data_list(negs)) if negs else None
        loss = (info_nce(za, zb, extra) + info_nce(zb, za, extra)) / 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"step {step}: loss {loss.item():.3f}")

    for split in ["val", "test"]:
        graphs = cd.load_pyg(args.track, split, positional = positional)
        emb = embed(model, graphs)
        np.savez(f"embeddings_{args.track}_{split}.npz", sample_ids = np.array(list(emb)), embeddings = np.stack(list(emb.values())))
        for condition in ["hard", "full"]:
            print(ev.evaluate(emb, args.track, split, condition))

def run_top20_cec_on_completion(track="real"):
    """Runs Top-20 retrieval and Yosys-ABC CEC verification on the validation embeddings."""
    npz_path = f"embeddings_{track}_val.npz"
    if not os.path.exists(npz_path):
        print(f"[Warning] {npz_path} not found. Skipping CEC evaluation.")
        return

    print(f"\n--- Running Top-20 CEC Verification ({track} track) ---")
    data = np.load(npz_path)
    embeddings = data["embeddings"]
    
    # Safely handle key name variations in the saved .npz file
    if "ids" in data:
        circuit_ids = data["ids"]
    elif "graph_ids" in data:
        circuit_ids = data["graph_ids"]
    elif "function_ids" in data:
        circuit_ids = data["function_ids"]
    else:
        circuit_ids = [f"circuit_{i}" for i in range(len(embeddings))]

    # Pick target query circuit (Index 0)
    query_id = circuit_ids[0]
    query_vec = embeddings[0].reshape(1, -1)

    # Compute cosine similarities
    sims = cosine_similarity(query_vec, embeddings)[0]
    top20_indices = np.argsort(sims)[::-1][:20]

    def find_bench_file(c_id):
        # Search recursively for the .bench file
        for root, dirs, files in os.walk("."):
            target_name = f"{c_id}.bench"
            if target_name in files:
                return os.path.join(root, target_name)
        return None

    query_bench_path = find_bench_file(query_id)

    def run_cec(path_a, path_b, timeout=60):
        if not path_a or not path_b or not os.path.exists(path_a) or not os.path.exists(path_b):
            return "Missing File"
        try:
            cmd = f"read {path_a}; read {path_b}; cec"
            res = subprocess.run(
                ["/opt/homebrew/bin/yosys-abc", "-c", cmd],
                capture_output = True,
                text = True,
                timeout = timeout
            )
            out = res.stdout + res.stderr
            if "Networks are equivalent" in out:
                return "Equivalent"
            elif "Networks are NOT EQUIVALENT" in out:
                return "Not Equivalent"
            return "Indeterminate"
        except subprocess.TimeoutExpired:
            return "Timeout (>60s)"
        except Exception:
            return "Error"

    print(f"Query Circuit: {query_id} (Path: {query_bench_path})")
    print(f"{'Rank':<5} {'Circuit ID':<35} {'Cosine Sim':<12} {'CEC Verification'}")
    print("-" * 70)

    for rank, idx in enumerate(top20_indices, 1):
        cand_id = circuit_ids[idx]
        score = sims[idx]
        cand_bench_path = find_bench_file(cand_id)
        
        status = run_cec(query_bench_path, cand_bench_path)
        print(f"{rank:<5} {str(cand_id):<35} {score:<12.4f} {status}")

if __name__ == "__main__":
    main()
    run_top20_cec_on_completion(track = "real")
