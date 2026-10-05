# eval metrics (Hit@1, P@k, MRR) + slot for GNN embeddings
import numpy as np
from .histogram import _cosine


def family(name):
    # 'c432_v3' -> 'c432'
    return name.split("_", 1)[0]


def evaluate(rank_fn, circuits, k=(5, 10), **kw):
    # rank every circuit against all the others and score the rankings
    names = list(circuits)
    hit1, rr, prec = [], [], {kk: [] for kk in k}
    for t in names:
        n_rel = sum(1 for c in names if c != t and family(c) == family(t))
        if n_rel == 0:
            continue
        hits = [family(c) == family(t) for c, _ in rank_fn(t, names, circuits, **kw)]
        hit1.append(float(hits[0]))
        rr.append(1 / (hits.index(True) + 1))
        for kk in k:
            prec[kk].append(sum(hits[:kk]) / min(kk, n_rel))
    if not hit1:
        return {}
    out = {"Hit@1": np.mean(hit1), **{f"P@{kk}": np.mean(v) for kk, v in prec.items()},
           "MRR": np.mean(rr)}
    return {m: round(float(v), 3) for m, v in out.items()}


def embedding_ranker(embeddings):
    # turns the GNN embeddings ({name: vector}) into a rank function, ranks by cosine
    E = {n: np.asarray(v, dtype=float).ravel() for n, v in embeddings.items()}

    def rank(target, candidates, circuits=None):
        out = [(c, _cosine(E[target], E[c])) for c in candidates if c != target]
        return sorted(out, key=lambda x: x[1], reverse=True)
    return rank
