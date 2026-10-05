# baseline 2 - WL kernel
import networkx as nx


def _wl_ready(G):
    # grakel only understands node labels, so each inverted edge gets an 'INV' node in the middle
    H = nx.Graph()
    for n, d in G.nodes(data=True):
        H.add_node(n, gate=d["gate"])
    for u, v, d in G.edges(data=True):
        if d.get("inverted"):
            mid = f"inv:{u}->{v}"
            H.add_node(mid, gate="INV")
            H.add_edge(u, mid); H.add_edge(mid, v)
        else:
            H.add_edge(u, v)
    return H


def wl_kernel(graphs, n_iter=3):
    # function 1: normalised WL kernel matrix (1.0 = identical structure)
    from grakel import WeisfeilerLehman, VertexHistogram
    from grakel.utils import graph_from_networkx
    gk = list(graph_from_networkx([_wl_ready(G) for G in graphs], node_labels_tag="gate"))
    return WeisfeilerLehman(n_iter=n_iter, base_graph_kernel=VertexHistogram,
                            normalize=True).fit_transform(gk)


_WL_CACHE = {}


def wl_rank(target, candidates, circuits, n_iter=3):
    # function 2: [(name, similarity), ...] most similar first
    key = (id(circuits), tuple(circuits), n_iter)
    if key not in _WL_CACHE:
        names = list(circuits)
        _WL_CACHE[key] = ({n: i for i, n in enumerate(names)},
                          wl_kernel([circuits[n] for n in names], n_iter))
    idx, K = _WL_CACHE[key]
    out = [(c, float(K[idx[target], idx[c]])) for c in candidates if c != target]
    return sorted(out, key=lambda x: x[1], reverse=True)
