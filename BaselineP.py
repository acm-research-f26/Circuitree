import re
import sys
import numpy as np 
import networkx as nx
from collections import Counter

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

    return np.array([gate_counts.get(gt, 0) for gt in GATE_TYPES], dtype = float)

def histogram_similarity(graph1, graph2):
    vec2 = gate_vector(graph2)
    vec1 = gate_vector(graph1)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def circuits_similarity_ranks(target_circuit, comparison_circuit):
    target_graph = load_bench(target_circuit)
    rankings = [
        (name, histogram_similarity(target_graph, load_bench(name)))
        for name in comparison_circuit
    ]

    rankings.sort(key = lambda x: x[1], reverse=True)
    return rankings


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "c17.bench"
    candiates = sys.argv[2] if len(sys.argv) > 2 else ["c17.bench"]
    
    rankings = circuits_similarity_ranks(target, candiates)
    for name, score in rankings:
        print(f"{name}: {score:.4f}")

if __name__ == "__main__":
    main()