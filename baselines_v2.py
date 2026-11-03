# baselines on dataset v2 - shireen
# histogram + WL, reads graphs/<track>.npz + samples.csv directly
import csv
import os
import time

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------- data
def load_split(root, track, split):
    # loads graphs + labels for one track/split
    # node_type: 0 = input, 1 = AND, 2 = output. edge_inv = 1 if the wire has a NOT on it
    rows = [r for r in csv.DictReader(open(os.path.join(root, "samples.csv"), newline=""))
            if r["track"] == track and r["split"] == split]
    with np.load(os.path.join(root, "graphs", f"{track}.npz")) as z:
        a = {k: z[k] for k in z.files}
    pos = {s: i for i, s in enumerate(a["sample_ids"].tolist())}
    idx = np.array([pos[r["sample_id"]] for r in rows])

    n_lo, n_hi = a["node_ptr"][idx], a["node_ptr"][idx + 1]
    e_lo, e_hi = a["edge_ptr"][idx], a["edge_ptr"][idx + 1]
    n_cnt, e_cnt = n_hi - n_lo, e_hi - e_lo
    node_sel = np.concatenate([np.arange(l, h) for l, h in zip(n_lo, n_hi)])
    edge_sel = np.concatenate([np.arange(l, h) for l, h in zip(e_lo, e_hi)])
    new_start = np.concatenate([[0], np.cumsum(n_cnt)[:-1]])
    edge_graph = np.repeat(np.arange(len(rows)), e_cnt)
    return dict(
        sample_id=[r["sample_id"] for r in rows],
        function_id=[r["function_id"] for r in rows],
        role=[r["role"] for r in rows],
        node_type=a["node_type"][node_sel].astype(np.int64),
        pi_index=a["pi_index"][node_sel].astype(np.int64),
        node_graph=np.repeat(np.arange(len(rows)), n_cnt),
        edge_src=a["edge_src"][edge_sel].astype(np.int64) + new_start[edge_graph],
        edge_dst=a["edge_dst"][edge_sel].astype(np.int64) + new_start[edge_graph],
        edge_inv=a["edge_inv"][edge_sel].astype(np.int64),
        edge_graph=edge_graph,
    )


# ---------------------------------------------------------------- baseline 1: histogram
def histogram_vectors(d):
    # counts [ANDs, inputs, outputs, normal wires, inverted wires] -> proportions
    # (NOTs are inverted wires in an AIG so this is basically the gate histogram)
    n = len(d["sample_id"])
    cnt = np.zeros((n, 5))
    for t in range(3):
        cnt[:, [1, 0, 2][t]] = np.bincount(d["node_graph"], weights=(d["node_type"] == t), minlength=n)
    inv = np.bincount(d["edge_graph"], weights=d["edge_inv"], minlength=n)
    cnt[:, 3] = np.bincount(d["edge_graph"], minlength=n) - inv
    cnt[:, 4] = inv
    return cnt / cnt.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- baseline 2: WL kernel
def _mix(x):
    # scrambles a number into a random-looking 64 bit number, used to make WL labels
    x = (x + np.uint64(0x9E3779B97F4A7C15))
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def wl_vectors(d, h=3, use_positions=False):
    # WL: start with node type, then each round a node's label = its label + its neighbours' labels
    # (neighbours tagged with direction + whether the wire is inverted)
    # count labels from every round -> vector. cosine of 2 vectors = normalized WL kernel
    # use_positions=True also tells it which input is x0, x1, ...
    with np.errstate(over="ignore"):
        lab = d["node_type"].copy()
        if use_positions:
            lab = np.where(lab == 0, 3 + d["pi_index"], lab)
        n_nodes, n = len(lab), len(d["sample_id"])
        src, dst, inv = d["edge_src"], d["edge_dst"], d["edge_inv"]
        blocks = []
        for it in range(h + 1):
            _, lab = np.unique(lab, return_inverse=True)            # renumber labels 0,1,2...
            blocks.append(sp.csr_matrix((np.ones(n_nodes), (d["node_graph"], lab)),
                                        shape=(n, lab.max() + 1)))
            if it == h:
                break
            L = lab.astype(np.uint64)
            # what each node hears from its inputs and its outputs
            m_in = _mix(L[src] * np.uint64(4) + inv.astype(np.uint64) * np.uint64(2))
            m_out = _mix(L[dst] * np.uint64(4) + inv.astype(np.uint64) * np.uint64(2) + np.uint64(1))
            acc = np.zeros(n_nodes, dtype=np.uint64)           # sum so order doesn't matter
            np.add.at(acc, dst, m_in)
            np.add.at(acc, src, m_out)
            # hash own label + neighbours separately (fixed a bug where 2 diff nodes got the same label)
            lab = _mix(_mix(L | np.uint64(1 << 62)) ^ _mix(acc))
    X = sp.hstack(blocks).tocsr()
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    return sp.diags(1 / np.maximum(norms, 1e-12)) @ X


# ---------------------------------------------------------------- scoring
def evaluate(root, track, split, vecs, sample_ids, condition="hard", seed=0):
    # same rules/metrics as the dataset's evaluate.py (checked, gives the same numbers)
    # but works with sparse vectors since the WL ones are huge
    rd = lambda name: list(csv.DictReader(open(os.path.join(root, name), newline="", encoding="utf-8")))
    samples = {r["sample_id"]: r for r in rd("samples.csv") if r["track"] == track and r["split"] == split}
    queries = rd(os.path.join("eval", f"{track}_{split}_queries.csv"))
    if condition == "hard":
        queries = [q for q in queries if q["in_hard_eval"] == "1"]
        gallery = {}
        for g in rd(os.path.join("eval", f"{track}_{split}_hard_gallery.csv")):
            gallery.setdefault(g["query_id"], []).append(g["candidate_id"])
    ids = list(sample_ids)
    index = {s: i for i, s in enumerate(ids)}
    M = sp.csr_matrix(vecs, dtype=np.float32) if sp.issparse(vecs) else np.asarray(vecs, dtype=np.float32)
    norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel()) if sp.issparse(M) else np.linalg.norm(M, axis=1)
    M = (sp.diags(1 / (norms + 1e-12)) @ M) if sp.issparse(M) else M / (norms[:, None] + 1e-12)
    fn_code = {f: i for i, f in enumerate(sorted({samples[s]["function_id"] for s in ids}))}
    pg_code = {g: i for i, g in enumerate(sorted({samples[s]["perm_group_id"] for s in ids} - {""}))}
    fn = np.array([fn_code[samples[s]["function_id"]] for s in ids])
    pg = np.array([pg_code.get(samples[s]["perm_group_id"], -1) for s in ids])
    q_rows = np.array([index[q["query_id"]] for q in queries], dtype=int)
    sims = M[q_rows] @ M.T
    sims = sims.toarray() if sp.issparse(sims) else np.asarray(sims)
    rng = np.random.default_rng(seed)
    res = []
    for qi, q in enumerate(queries):
        r = q_rows[qi]
        cand = (np.array([index[c] for c in gallery[q["query_id"]]], dtype=int) if condition == "hard"
                else np.delete(np.arange(len(ids)), r))
        same = fn[cand] == fn[r]
        keep = same | (pg[cand] == -1) | (pg[cand] != pg[r])
        s, lab = sims[qi, cand[keep]], same[keep]
        order = np.lexsort((rng.random(len(s)), -s))
        hits = lab[order]
        p = np.flatnonzero(hits)
        if len(p) == 0:
            continue
        res.append(dict(hit1=float(hits[0]), mrr=1 / (p[0] + 1), r5=hits[:5].sum() / len(p),
                        r10=hits[:10].sum() / len(p), ap=float((np.arange(1, len(p) + 1) / (p + 1)).mean())))
    out = {k: round(float(np.mean([x[k] for x in res])), 3) for k in ["hit1", "mrr", "r5", "r10", "ap"]}
    out["queries"] = len(res)
    return out


def timed(fn, *args, **kw):
    t = time.perf_counter()
    out = fn(*args, **kw)
    return out, time.perf_counter() - t
