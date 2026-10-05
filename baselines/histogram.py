# baseline 1 - histogram
import numpy as np




def _cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def histogram_vector(G):
    # function 1: normalised counts of [AND, PI, PO, normal edges, inverted edges]
    gates = [d["gate"] for _, d in G.nodes(data=True)]
    inv = sum(1 for *_, d in G.edges(data=True) if d.get("inverted"))
    v = np.array([gates.count("AND"), gates.count("INPUT"), gates.count("OUTPUT"),
                  G.number_of_edges() - inv, inv], dtype=float)
    return v / v.sum() if v.sum() else v


def histogram_rank(target, candidates, circuits):
    # function 2: [(name, similarity), ...] most similar first (cosine)
    t = histogram_vector(circuits[target])
    out = [(c, _cosine(t, histogram_vector(circuits[c]))) for c in candidates if c != target]
    return sorted(out, key=lambda x: x[1], reverse=True)
