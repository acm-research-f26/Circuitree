"""
fgnn2_torch.py -- FGNN2 fingerprints using ONLY PyTorch (no DGL needed).

Same math as FGNN2's FunctionConv.forward_global, rewritten without DGL so it runs anywhere
PyTorch runs (Kaggle, Windows, ...). Loads the same weight files (pretrained, Exp A, B, C).

For every topological level after the inputs, for every gate in that level:
    message from each input wire = h[source], passed through func_inv if the wire has a NOT
    neigh = average of the messages
    h[gate] = func_and(neigh)
Circuit fingerprint = concat(mean, max) over the output nodes (NOT outputs go through func_inv),
exactly like embed_circuitree.py.

Example:
    python fgnn2_torch.py --data_root DATA --weights expB_best.pth --positions --out B_val.npz
"""
import argparse, csv, os, sys, time
import numpy as np
import torch as th
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_retrieval import parse_bench          # pure Python, no DGL


class MLP(nn.Module):                           # identical to FunctionConv.MLP (same weight names)
    def __init__(self, *sizes, negative_slope=0.01):
        super().__init__()
        fcs = []
        for i in range(1, len(sizes)):
            fcs.append(nn.Linear(sizes[i - 1], sizes[i]))
            if i < len(sizes) - 1:
                fcs.append(nn.LeakyReLU(negative_slope=negative_slope))
        self.layers = nn.Sequential(*fcs)

    def forward(self, x):
        return self.layers(x)


class FGNN2(nn.Module):
    def __init__(self, hidden=256, positions=False, max_inputs=256):
        super().__init__()
        self.hidden = hidden
        self.func_inv = MLP(hidden, hidden // 2, hidden // 2, hidden)
        self.func_and = MLP(hidden, hidden // 2, hidden // 2, hidden)
        self.proj_head = MLP(hidden, hidden, hidden, negative_slope=0)    # unused, kept so weights load
        self.pos_emb = nn.Embedding(max_inputs + 1, hidden, padding_idx=0) if positions else None
        if positions:
            nn.init.zeros_(self.pos_emb.weight)

    def load(self, path):
        if not path:
            return
        state = th.load(path, map_location='cpu', weights_only=True)
        missing = [k for k in self.state_dict() if k not in state]
        self.load_state_dict(state, strict=not missing)
        if missing:
            print('note: these parts were not in the weight file and start at zero:', sorted({k.split('.')[0] for k in missing}))


def levels(p):
    """Topological level of every node (inputs = 0), same frontiers as dgl.topological_nodes_generator."""
    n = p['n_nodes']
    fanout = [[] for _ in range(n)]
    indeg = np.zeros(n, dtype=np.int64)
    for s, d in zip(p['src'], p['dst']):
        fanout[s].append(d); indeg[d] += 1
    lvl = np.zeros(n, dtype=np.int64)
    frontier = [i for i in range(n) if indeg[i] == 0]
    k = 0
    while frontier:
        nxt = []
        for u in frontier:
            lvl[u] = k
            for v in fanout[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    nxt.append(v)
        frontier = nxt; k += 1
    return lvl


def embed_core(model, parsed, device='cpu', pos_lists=None):
    """Fingerprints for a list of parsed circuits, all in one batch -> tensor [len(parsed), 2*hidden].
    Works with gradients (training) or inside torch.no_grad() (scoring).
    pos_lists[k] (optional) = custom input positions for circuit k (default 1, 2, 3, ...)."""
    offs = np.cumsum([0] + [p['n_nodes'] for p in parsed])
    N = int(offs[-1])
    src = th.tensor(np.concatenate([np.asarray(p['src'], dtype=np.int64) + o for p, o in zip(parsed, offs)]), device=device)
    dst = th.tensor(np.concatenate([np.asarray(p['dst'], dtype=np.int64) + o for p, o in zip(parsed, offs)]), device=device)
    r = th.tensor(np.concatenate([np.asarray(p['r'], dtype=np.int64) for p in parsed]), device=device).bool()
    lvl = th.tensor(np.concatenate([p['lvl'] if 'lvl' in p else levels(p) for p in parsed]), device=device)

    h = th.ones(N, model.hidden, device=device)
    if model.pos_emb is not None:
        pos = th.zeros(N, dtype=th.long)
        for k, (p, o) in enumerate(zip(parsed, offs)):
            if pos_lists is not None and pos_lists[k] is not None:
                pos[o:o + p['n_pi']] = th.tensor(pos_lists[k], dtype=th.long)
            else:
                pos[o:o + p['n_pi']] = th.arange(1, p['n_pi'] + 1)
        pos = pos.clamp(max=model.pos_emb.num_embeddings - 1).to(device)
        h = h + model.pos_emb(pos)

    edge_lvl = lvl[dst]
    for k in range(1, int(lvl.max()) + 1):
        e = (edge_lvl == k).nonzero(as_tuple=True)[0]
        s, d, inv = src[e], dst[e], r[e]
        msg = h[s].clone()
        if inv.any():
            msg[inv] = model.func_inv(msg[inv])
        agg = th.zeros(N, model.hidden, device=device).index_add(0, d, msg)
        cnt = th.zeros(N, device=device).index_add(0, d, th.ones(len(d), device=device))
        nodes = (lvl == k).nonzero(as_tuple=True)[0]
        h = h.index_put((nodes,), model.func_and(agg[nodes] / cnt[nodes].clamp(min=1)[:, None]))

    out = []
    for p, o in zip(parsed, offs):
        po = th.tensor(p['po_nids'], dtype=th.long, device=device) + int(o)
        ph = h[po]
        pinv = th.tensor(p['po_inv'], dtype=th.bool, device=device)
        if pinv.any():
            ph = th.where(pinv[:, None], model.func_inv(ph), ph)
        out.append(th.cat([ph.mean(0), ph.max(0).values]))
    return th.stack(out)


@th.no_grad()
def embed(model, parsed, device='cpu', pos_lists=None):
    return embed_core(model, parsed, device, pos_lists).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--track', default='real')
    ap.add_argument('--split', default='val')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--positions', action='store_true')
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    device = 'cuda' if th.cuda.is_available() else 'cpu'
    model = FGNN2(positions=a.positions)
    model.load(a.weights)
    model.to(device).eval()

    with open(os.path.join(a.data_root, 'samples.csv'), newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['track'] == a.track and r['split'] == a.split]
    rows.sort(key=lambda r: int(r['depth']))
    t = time.perf_counter()
    parsed = [parse_bench(os.path.join(a.data_root, r['file'])) for r in rows]
    print(f'{len(rows)} circuits read in {time.perf_counter() - t:.0f} s')

    t = time.perf_counter()
    embs = []
    for s in range(0, len(parsed), a.batch_size):
        embs.append(embed(model, parsed[s:s + a.batch_size], device))
    dt = time.perf_counter() - t
    np.savez(a.out, sample_ids=np.array([r['sample_id'] for r in rows]), embeddings=np.concatenate(embs))
    print(f'saved {a.out}  (embedding took {dt:.0f} s, {1000 * dt / len(rows):.1f} ms per circuit on CPU)')


if __name__ == '__main__':
    main()
