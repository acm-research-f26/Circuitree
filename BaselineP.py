import re
import sys
from collections import Counter

import networkx as nx

GATE_RE = re.compile(r"^(\S+)\s*=\s*(\w+)\s*\((.*)\)$")

IO_RE = re.compile(r"^(INPUT|OUTPUT)\s*\(\s*(\S+?)\s*\)$", re.IGNORECASE)
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

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "c17.bench"
    g = load_bench(path)

    gate_counts = Counter(
        d["gate"] for _, d in g.nodes(data = True)
        if d.get("gate") not in (None, "INPUT")
    )

    print(f"Nodes: {g.number_of_nodes()}")
    print(f"Edges: {g.number_of_edges()}")
    for gate_type, count in sorted(gate_counts.items()):
        print(f"{gate_type} : {count}")

if __name__ == "__main__":
    main()