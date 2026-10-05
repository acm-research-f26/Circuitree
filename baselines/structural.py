# baseline 3 - structural [#nodes, #edges, depth, #PI, #PO]
import numpy as np
from .aig_loader import aig_stats


def structural_vector(G):
    # function 1: log of [#nodes, #edges, depth, #PI, #PO]
    s = aig_stats(G)
    return np.log1p([s["nodes"], s["edges"], s["depth"], s["PI"], s["PO"]])


def structural_rank(target, candidates, circuits):
    # function 2: [(name, similarity), ...] most similar first
    t = structural_vector(circuits[target])
    out = [(c, 1 / (1 + float(np.linalg.norm(t - structural_vector(circuits[c])))))
           for c in candidates if c != target]
    return sorted(out, key=lambda x: x[1], reverse=True)
