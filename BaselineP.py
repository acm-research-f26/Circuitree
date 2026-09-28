import re
import sys
import numpy as np 
import networkx as nx
from collections import Counter
from grakel import WeisfeilerLehman, VertexHistogram
from grakel.utils import graph_from_networkx

GATE_RE = re.compile(r"^(\S+)\s*=\s*(\w+)\s*\((.*)\)$")
IO_RE = re.compile(r"^(INPUT|OUTPUT)\s*\(\s*(\S+?)\s*\)$", re.IGNORECASE)

GATE_TYPES = sorted(["AND", "NAND", "OR", "NOR", "NOT", "XOR", "XNOR", "BUF"])

def load_bench(path):
    g = nx.DiGraph()
    with open(path) as f:
        for raw in f:
            line = raw.split("#")[0].strip()
            if not line:
                continue 

            m = IO_RE.match(line)
            if m:
                kind, name = m.group(1).upper(), m.group(2)
                if kind == "INPUT":
                    g.add_node(name, gate = "INPUT")
                else : 
                    g.add_node(name, is_output = True)
                continue 

            m = GATE_RE.match(line)
            if m:
                out, gate_type, args = m.groups()
                g.add_node(out, gate=gate_type.upper())
                for src in (a.strip() for a in args.split(",")):
                    g.add_edge(src, out)
    return g 

def gate_vector(graph):
    gate_counts = Counter(
        d["gate"] for _, d in graph.nodes(data=True)
        if d.get("gate") not in (None, "INPUT")
    )
    known = {gt: gate_counts.get(gt, 0) for gt in GATE_TYPES if gt != "OTHER"}
    other = sum(c for gt, c in gate_counts.items() if gt not in GATE_TYPES)
    known["OTHER"] = other

    return np.array([known[gt] for gt in GATE_TYPES], dtype = float)

def histogram_similarity(graph1, graph2):
    vec2 = gate_vector(graph2)
    vec1 = gate_vector(graph1)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def circuits_similarity_ranking(target_circuit, comparison_circuit):
    target_graph = load_bench(target_circuit)
    rankings = [
        (name, histogram_similarity(target_graph, load_bench(name)))
        for name in comparison_circuit
    ]

    rankings.sort(key = lambda x: x[1], reverse=True)
    return rankings


def grakel_initialization(nx_graph):
    g = nx_graph.copy()
    for node, data in g.nodes(data=True):
        if "gate" not in data:
            data["gate"] = "OUTPUT"
    return g

def WL_similarity(graph1, graph2, h = 3):
    g1 = grakel_initialization(graph1)
    g2 = grakel_initialization(graph2)

    grakel_graphs = list(graph_from_networkx([g1,g2], node_labels_tag = "gate"))

    gk = WeisfeilerLehman(n_iter = h, base_graph_kernel = VertexHistogram, normalize = True)
    K = gk.fit_transform(grakel_graphs)

    return float(K[0, 1])

def ciruits_similarity_ranking_WL(target_circuit, compariosn_ciruit, h = 3):
    target_graph = load_bench(target_circuit)
    rankings= [
        (name, WL_similarity(target_graph, load_bench(name), h = h))
        for name in compariosn_ciruit
    ]
    rankings.sort(key = lambda x: x[1], reverse = True)
    return rankings


def main():
    import glob
    all_circuits = glob.glob("*.bench")
    if "c17.bench" in all_circuits:
        all_circuits.remove("c17.bench")

    print("Histogram:")
    rankings = circuits_similarity_ranking("c17.bench", ["c17.bench"] + all_circuits)
    for name, score in rankings:
        print(f"{name}: {score:.4f}")

    print("WL:")
    wl_rankings = ciruits_similarity_ranking_WL("c17.bench", ["c17.bench"] + all_circuits, h=3)
    for name, score in wl_rankings:
        print(f"{name}: {score:.4f}")


if __name__ == "__main__":
    main()