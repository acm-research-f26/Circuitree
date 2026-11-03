import argparse, os, random, shutil, subprocess, sys, time
import numpy as np, torch, torch.nn.functional as F
from torch_geometric.data import Batch

sys.path.insert(0, os.path.abspath("../.."))      # Model.py lives in Circuitree/
from Model import CircuitGNN, InfoNCELoss         # YOUR model and loss
import circuitree_data as cd
import evaluate as ev

def encode(model, b):
    return model(b.x, b.edge_index, b.edge_type, b.batch)

def embed(model, graphs, bs = 256):               # batched + no_grad: avoids the >800 circuit crash
    model.eval(); out = {}
    with torch.no_grad():
        for i in range(0, len(graphs), bs):
            chunk = graphs[i:i + bs]
            z = F.normalize(encode(model, Batch.from_data_list(chunk)), dim = 1)
            for d, v in zip(chunk, z):
                out[d.sample_id] = v.numpy()
    return out

def hit1_within(emb, fid):
    """Nearest-neighbour hit rate inside one set. Same method on train and val subsets."""
    ids = list(emb); M = np.stack([emb[i] for i in ids]); S = M @ M.T
    np.fill_diagonal(S, -2)
    f = np.array([fid[i] for i in ids])
    counts = {x: (f == x).sum() for x in set(f)}
    ok = [i for i in range(len(ids)) if counts[f[i]] > 1]       # queries that have a true match
    return float(np.mean([f[S[i].argmax()] == f[i] for i in ok])), len(ok)

def run_cec(a, b, timeout = 900): # 15 minutes worth of timeout for real, more complex circuits 
    exe = shutil.which("yosys-abc") or "/opt/homebrew/bin/yosys-abc"
    try:
        r = subprocess.run([exe, "-c", f"cec {a} {b}"], capture_output = True, text = True, timeout = timeout)
        out = r.stdout + r.stderr
        if "Networks are equivalent" in out: 
            return "Equivalent"
        else:
            return "Not Equivalent"
        # if "NOT EQUIVALENT" in out.upper(): 
        #     return "Not equivalent"
        # return "Indeterminate"
    except subprocess.TimeoutExpired:
        return "Timeout" # Added so that there is no false negatives
    except Exception: 
        return "Error" # Added so aid in debugging but also to prevent false negatives

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", default = "synthetic", choices = ["real", "synthetic"])
    p.add_argument("--steps", type = int, default = 500)
    p.add_argument("--batch-functions", type = int, default = 64)
    p.add_argument("--near-miss", action = "store_true")
    p.add_argument("--target", default = None, help = "sample_id of the query circuit")
    a = p.parse_args(); random.seed(0); torch.manual_seed(0)

    train = cd.load_pyg(a.track, "train", positional = True)
    rows = {r["sample_id"]: r for r in cd.read_samples(track = a.track, split = "train")}
    by_fn, mutants = {}, {}
    for d in train:
        r = rows[d.sample_id]
        if r["role"] == "view": by_fn.setdefault(d.function_id, []).append(d)
        else: mutants.setdefault(r["source_function_id"], []).append(d)
    fns = [f for f, v in by_fn.items() if len(v) >= 2]
    print(f"train: {len(train)} graphs, {len(fns)} functions")

    model = CircuitGNN(in_channels = train[0].x.size(1), hidden_channels = 64, out_channels = 64, num_layers = 3)
    loss_fn = InfoNCELoss(0.07); opt = torch.optim.Adam(model.parameters(), lr = 1e-3)
    t0, window = time.time(), []
    for step in range(1, a.steps + 1):
        model.train()
        bf = random.sample(fns, min(a.batch_functions, len(fns)))
        pairs = [random.sample(by_fn[f], 2) for f in bf]
        za = encode(model, Batch.from_data_list([x for x, _ in pairs]))
        zp = encode(model, Batch.from_data_list([y for _, y in pairs]))
        extra = None
        if a.near_miss:
            negs = [random.choice(mutants[f]) for f in bf if f in mutants]
            extra = encode(model, Batch.from_data_list(negs)) if negs else None
        loss = loss_fn(za, zp, extra)
        opt.zero_grad(); loss.backward(); opt.step(); window.append(loss.item())
        if step % 100 == 0:
            print(f"step {step}: avg loss (last 100) {np.mean(window):.3f}  [{time.time()-t0:.0f}s]"); window = []
    print(f"[Metrics] training time: {time.time()-t0:.1f}s")
    torch.save(model.state_dict(), f"rgcn_{a.track}.pt")

    # ---- accuracy on val/test with the lead's scorer ----
    val_graphs = None
    for split in ["val", "test"]:
        graphs = cd.load_pyg(a.track, split, positional = True)
        if split == "val": val_graphs = graphs
        emb = embed(model, graphs)
        np.savez(f"embeddings_{a.track}_{split}.npz", sample_ids = np.array(list(emb)),
                 embeddings = np.stack(list(emb.values())))
        for cond in ["hard", "full"]:
            print(split, cond, ev.evaluate(emb, a.track, split, cond))

    # ---- memorization check: same retrieval test on train vs unseen val subsets ----
    n = min(2000, len(train), len(val_graphs))
    tr_sub, va_sub = random.sample(train, n), random.sample(val_graphs, n)
    tr_h, tr_q = hit1_within(embed(model, tr_sub), {d.sample_id: d.function_id for d in tr_sub})
    va_h, va_q = hit1_within(embed(model, va_sub), {d.sample_id: d.function_id for d in va_sub})
    print(f"[Memorization] Hit@1 train subset {tr_h:.3f} ({tr_q} queries) vs val subset {va_h:.3f} ({va_q} queries)")
    print("  big train>>val gap = memorizing; similar = generalizing")

    # ---- top-20 + CEC on a val target ----
    vrows = {r["sample_id"]: r for r in cd.read_samples(track = a.track, split = "val")}
    emb = embed(model, val_graphs); ids = list(emb); M = np.stack([emb[i] for i in ids])
    target = a.target or next(i for i in ids if vrows[i]["role"] == "view")
    target_idx = ids.index(target)
    sims = M @ emb[target]; order = [i for i in np.argsort(-sims) if ids[i] != target_idx][:20]
    qpath = f"circuits/{a.track}/{target}.bench"
    print(f"\nTarget {target}  function {vrows[target]['function_id']}")
    print(f"{'Rank':<5}{'Circuit':<12}{'Cosine':<9}{'Same function (label)':<23}CEC")
    for rank, i in enumerate(order, 1):
        cid = ids[i]; cpath = f"circuits/{a.track}/{cid}.bench"
        cec = run_cec(qpath, cpath) if os.path.exists(cpath) else "Missing file"
        same = vrows[cid]["function_id"] == vrows[target]["function_id"]
        print(f"{rank:<5}{cid:<12}{sims[i]:<9.4f}{str(same):<23}{cec}")

if __name__ == "__main__":
    main()