"""
eval_retrieval.py  --  measure FGNN2 on circuit retrieval
Metrics: retrieval@k (hit@k), MRR, top-1 accuracy, and timing.

Put this file in FGNN2/src/ (it imports FunctionConv.py).

Expected data layout (one folder per source circuit, variants inside):
    bench_root/
        c432/   c432_v0.bench  c432_v1.bench ...   <- all variants of c432
        c880/   c880_v0.bench  ...
Two circuits "match" if they come from the same folder.

Example:
    python eval_retrieval.py --bench_root ../data/iscas_variants \
        --weights ../checkpoints/example/weight.pth --flag_inv --ks 1 5 10
"""
import argparse, os, time, json
import numpy as np


# ---------------------------------------------------------------- parsing
def parse_bench(path):
    """Parse an ABC-style AIG .bench (INPUT/OUTPUT/AND/NOT/BUFF, multi-output).
    NOT gates are folded into inverted edges (edge flag r=1), same idea as
    pretrain_dataparser.py, but this version supports many outputs."""
    inputs, outputs, defs = [], [], {}
    with open(path) as f:
        for raw in f:
            line = raw.split('#')[0].strip()
            if not line:
                continue
            up = line.upper()
            if up.startswith('INPUT('):
                inputs.append(line[line.find('(') + 1:line.rfind(')')].strip())
            elif up.startswith('OUTPUT('):
                outputs.append(line[line.find('(') + 1:line.rfind(')')].strip())
            elif '=' in line:
                lhs, rhs = [s.strip() for s in line.split('=', 1)]
                if '(' in rhs:
                    gate = rhs[:rhs.find('(')].strip().upper()
                    args = [a.strip() for a in rhs[rhs.find('(') + 1:rhs.rfind(')')].split(',') if a.strip()]
                else:                      # e.g. "n0 = vdd" / "n0 = gnd"
                    gate, args = rhs.strip().upper(), []
                defs[lhs] = (gate, args)

    def resolve(sig, depth=0):
        """Follow NOT/BUFF chains -> (real node name, inverted?)"""
        inv = False
        while sig in defs and defs[sig][0] in ('NOT', 'BUFF', 'BUF'):
            g, a = defs[sig]
            if g == 'NOT':
                inv = not inv
            sig = a[0]
        return sig, inv

    nid = {}
    def get_id(name):
        if name not in nid:
            nid[name] = len(nid)
        return nid[name]

    for pi in inputs:
        get_id(pi)
    src, dst, r = [], [], []
    for name, (gate, args) in defs.items():
        if gate == 'AND':
            d = get_id(name)
            for a in args:
                s, inv = resolve(a)
                src.append(get_id(s)); dst.append(d); r.append(int(inv))
        elif gate in ('NOT', 'BUFF', 'BUF'):
            continue
        elif gate in ('VDD', 'GND', 'CONST0', 'CONST1'):
            get_id(name)              # constant: source node, like a PI
        else:
            raise ValueError(f'{path}: gate {gate} is not AIG. Run "strash" in ABC first.')

    po_nids, po_inv = [], []
    for po in outputs:
        s, inv = resolve(po)
        po_nids.append(get_id(s)); po_inv.append(int(inv))

    return dict(n_nodes=len(nid), src=src, dst=dst, r=r,
                po_nids=po_nids, po_inv=po_inv,
                n_pi=len(inputs), n_and=sum(1 for g, _ in defs.values() if g == 'AND'))


def build_graph(p):
    import torch as th, dgl
    g = dgl.graph((th.tensor(p['src'], dtype=th.long), th.tensor(p['dst'], dtype=th.long)),
                  num_nodes=p['n_nodes'])
    g.edata['r'] = th.tensor(p['r'], dtype=th.long)
    g.ndata['inv'] = th.zeros(p['n_nodes'])      # PO inversion handled after the GNN
    return g


# ---------------------------------------------------------------- embedding
def embed_batch(model, graphs, po_lists, po_inv_lists, device):
    """Embed many circuits in ONE forward pass (dgl.batch).
    Returns [num_graphs, 2*hidden] = concat(mean, max) over PO embeddings,
    the same readout train_global.py uses."""
    import torch as th, dgl
    bg = dgl.batch(graphs).to(device)
    bg.ndata['h'] = th.ones((bg.num_nodes(), model.hidden_dim), device=device)
    topo = [l.to(device) for l in dgl.topological_nodes_generator(bg)]
    use_proj = model.flag_proj
    model.set_flag_proj(False)                                # project only after PO inversion
    h = model(bg, topo, None, flag_usage='global')            # all node embeddings
    model.set_flag_proj(use_proj)

    offs = th.cumsum(th.tensor([0] + [g.num_nodes() for g in graphs[:-1]]), 0)
    po_idx, seg, inv = [], [], []
    for k, (pl, il) in enumerate(zip(po_lists, po_inv_lists)):
        po_idx += [int(offs[k]) + p for p in pl]; seg += [k] * len(pl); inv += il
    po_idx = th.tensor(po_idx, device=device); seg = th.tensor(seg, device=device)
    inv = th.tensor(inv, device=device, dtype=th.bool)

    po_h = h[po_idx].clone()
    if model.flag_inv and inv.any():
        po_h[inv] = model.func_inv(po_h[inv])                 # inverted outputs
    if use_proj:
        po_h = model.proj_head(po_h)

    B, D = len(graphs), po_h.shape[1]
    cnt = th.zeros(B, device=device).index_add_(0, seg, th.ones_like(seg, dtype=th.float))
    mean = th.zeros(B, D, device=device).index_add_(0, seg, po_h) / cnt.clamp(min=1)[:, None]
    mx = th.full((B, D), -1e9, device=device).scatter_reduce(
        0, seg[:, None].expand(-1, D), po_h, reduce='amax', include_self=True)
    return th.cat([mean, mx], 1)


# ---------------------------------------------------------------- metrics
def retrieval_metrics(emb, groups, ks=(1, 5, 10)):
    """Leave-one-out retrieval. Every circuit is a query; the rest is the gallery.
    hit@k = was at least one same-group circuit in the top k?
    MRR   = average of 1/(rank of first same-group circuit)
    acc   = top-1 nearest neighbour has the right group (== hit@1)"""
    emb = np.asarray(emb, dtype=np.float64)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    groups = np.asarray(groups)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)
    rel = groups[order] == groups[:, None]
    has = rel.any(1)                                  # skip groups of size 1
    first = rel[has].argmax(1) + 1                    # 1-based rank
    out = {f'hit@{k}': float((first <= k).mean()) for k in ks}
    out['MRR'] = float((1.0 / first).mean())
    out['top1_acc'] = out.get('hit@1', float((first == 1).mean()))
    out['mean_rank'] = float(first.mean())
    out['num_queries'] = int(has.sum())
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench_root', required=True)
    ap.add_argument('--weights', default=None, help='pretrained FuncConv state_dict (.pth)')
    ap.add_argument('--hidden_dim', type=int, default=256)
    ap.add_argument('--out_dim', type=int, default=256)
    ap.add_argument('--flag_inv', action='store_true')
    ap.add_argument('--flag_proj', action='store_true')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--ks', type=int, nargs='+', default=[1, 5, 10])
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out_json', default=None)
    ap.add_argument('--show_misses', action='store_true',
                    help='print every circuit whose #1 pick was not a twin')
    a = ap.parse_args()

    import torch as th
    from FunctionConv import FuncConv
    device = th.device(f'cuda:{a.gpu}' if th.cuda.is_available() else 'cpu')
    sync = (lambda: th.cuda.synchronize()) if device.type == 'cuda' else (lambda: None)

    model = FuncConv(a.hidden_dim, a.out_dim, flag_proj=a.flag_proj, flag_inv=a.flag_inv).to(device)
    if a.weights:
        model.load_state_dict(th.load(a.weights, map_location=device, weights_only=True))
    else:
        print('WARNING: no --weights, using a random untrained model (useful as a baseline)')
    model.eval()

    # 1) parse
    t0 = time.perf_counter()
    items = []
    for grp in sorted(os.listdir(a.bench_root)):
        gdir = os.path.join(a.bench_root, grp)
        if not os.path.isdir(gdir):
            continue
        for fn in sorted(os.listdir(gdir)):
            if fn.endswith('.bench'):
                items.append((grp, fn, parse_bench(os.path.join(gdir, fn))))
    t_parse = time.perf_counter() - t0
    print(f'parsed {len(items)} circuits from {len({i[0] for i in items})} groups')

    # 2) build graphs
    t0 = time.perf_counter()
    graphs = [build_graph(p) for _, _, p in items]
    t_build = time.perf_counter() - t0

    # 3) embed (timed)
    embs = []
    with th.no_grad():
        embed_batch(model, graphs[:2], [p['po_nids'] for *_, p in items[:2]],
                    [p['po_inv'] for *_, p in items[:2]], device)      # warm-up
        sync(); t0 = time.perf_counter()
        for s in range(0, len(graphs), a.batch_size):
            chunk = items[s:s + a.batch_size]
            embs.append(embed_batch(model, graphs[s:s + a.batch_size],
                                    [p['po_nids'] for *_, p in chunk],
                                    [p['po_inv'] for *_, p in chunk], device).cpu())
        sync(); t_embed = time.perf_counter() - t0
    emb = th.cat(embs).numpy()

    # 4) search + metrics (timed)
    t0 = time.perf_counter()
    res = retrieval_metrics(emb, [g for g, _, _ in items], a.ks)
    t_search = time.perf_counter() - t0

    n = len(items)
    res.update({
        'num_circuits': n, 'device': str(device),
        'time_parse_s': t_parse, 'time_build_s': t_build,
        'time_embed_s': t_embed, 'ms_per_circuit_embed': 1000 * t_embed / n,
        'time_search_all_queries_s': t_search,
        'avg_and_nodes': float(np.mean([p['n_and'] for *_, p in items])),
    })
    for k, v in res.items():
        print(f'{k:>28}: {v:.4f}' if isinstance(v, float) else f'{k:>28}: {v}')
    if a.show_misses:
        e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        sim = e @ e.T
        np.fill_diagonal(sim, -np.inf)
        top = sim.argmax(1)
        print('\nMISSES (circuit -> its #1 pick, which was NOT a twin):')
        found = False
        for i in range(n):
            if items[top[i]][0] != items[i][0]:
                found = True
                print(f'  {items[i][1]:>20} -> {items[top[i]][1]:<20} similarity {sim[i, top[i]]:.4f}')
        if not found:
            print('  none!')
    if a.out_json:
        with open(a.out_json, 'w') as f:
            json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
