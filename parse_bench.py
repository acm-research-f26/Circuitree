"""
Circuitree Build Night 1

How to run:
    python parse_bench.py c17.bench
"""

import sys
import re
from collections import Counter

import networkx as nx


def parse_bench(filepath):
    """
    Parse a .bench file into a networkx.DiGraph.

    Returns:
        graph (networkx.DiGraph), gate_counts (Counter of gate type -> count)
    """
    graph = nx.DiGraph()
    gate_counts = Counter()

    # Matches lines like 22 = NAND(10, 16)
    gate_pattern = re.compile(r"^\s*(\S+)\s*=\s*(\w+)\((.*)\)\s*$")

    with open(filepath, "r") as f:
        for raw_line in f:
            line = raw_line.strip()

            # Skip blank lines and comments
            if not line or line.startswith("#"):
                continue

            if line.startswith("INPUT("):
                signal = line[len("INPUT("):-1].strip()
                graph.add_node(signal, type="input")
                continue

            if line.startswith("OUTPUT("):
                signal = line[len("OUTPUT("):-1].strip()
                # OUTPUT lines just mark an existing signal as a circuit
                if graph.has_node(signal):
                    graph.nodes[signal]["is_output"] = True
                else:
                    graph.add_node(signal, is_output=True)
                continue

            match = gate_pattern.match(line)
            if match:
                out_signal, gate_type, args = match.groups()
                in_signals = [a.strip() for a in args.split(",") if a.strip()]

                gate_counts[gate_type] += 1

                graph.add_node(out_signal, type="gate", gate=gate_type)
                for in_signal in in_signals:
                    if not graph.has_node(in_signal):
                        graph.add_node(in_signal)
                    graph.add_edge(in_signal, out_signal)
                continue

            # Anything unrecognized — flag it rather than just skipping
            print(f"Warning: could not parse line: {line!r}")

    return graph, gate_counts


def main():
    if len(sys.argv) != 2:
        print("Usage: python parse_bench.py <path-to-bench-file>")
        sys.exit(1)

    filepath = sys.argv[1]
    graph, gate_counts = parse_bench(filepath)

    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print("Gate counts:")
    for gate_type, count in gate_counts.items():
        print(f"  {gate_type}: {count}")


if __name__ == "__main__":
    main()