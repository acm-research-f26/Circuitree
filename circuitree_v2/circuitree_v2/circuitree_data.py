"""Helpers for reading the Circuitree dataset.

The .bench files under circuits/ are the primary data; samples.csv holds every label.
Nothing here is required: use it as a starting point or write your own reader.

    from circuitree_data import read_samples, parse_bench, bench_to_networkx
    samples = read_samples(".", track="real", split="train")
    circuit = parse_bench(samples[0]["file"])

Optional extras need numpy (load_graph_arrays), networkx (bench_to_networkx) or
torch + torch_geometric (to_pyg / load_pyg).
"""
import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
INT_FIELDS = {"num_inputs", "and_count", "depth", "edge_count", "num_views", "num_near_miss"}


# ---------------------------------------------------------------- labels (stdlib only)
def read_csv(name, root=HERE):
    with open(os.path.join(root, name), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in INT_FIELDS & r.keys():
            r[k] = int(r[k])
    return rows


def read_samples(root=HERE, track=None, split=None, role=None):
    """Rows of samples.csv, optionally filtered. 'file' is made an absolute path."""
    rows = read_csv("samples.csv", root)
    rows = [r for r in rows if (track is None or r["track"] == track) and (split is None or r["split"] == split)
            and (role is None or r["role"] == role)]
    for r in rows:
        r["file"] = os.path.join(root, r["file"])
    return rows


# ---------------------------------------------------------------- .bench parsing (stdlib only)
def parse_bench(path):
    """Parse a dataset .bench file.

    Returns dict(inputs=[x0, x1, ...] in positional order, output='y0',
                 gates={signal: (op, [fan-in signals])} in file (topological) order).
    Ops are AND (2 inputs), NOT, and BUFF (only when the output is wired to an input).
    """
    inputs, outputs, gates = [], [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            m = re.fullmatch(r"(INPUT|OUTPUT)\((\w+)\)", line)
            if m:
                (inputs if m[1] == "INPUT" else outputs).append(m[2])
                continue
            m = re.fullmatch(r"(\w+)\s*=\s*(\w+)\(([^)]*)\)", line)
            if not m:
                raise ValueError(f"cannot parse line: {line}")
            gates[m[1]] = (m[2], [a.strip() for a in m[3].split(",")])
    return dict(inputs=inputs, output=outputs[0], gates=gates)


def simulate(circuit, assignment):
    """Output bit for one input assignment (a sequence of 0/1, one per input in order)."""
    value = dict(zip(circuit["inputs"], assignment))
    for s, (op, args) in circuit["gates"].items():
        if op == "AND":
            value[s] = value[args[0]] & value[args[1]]
        elif op == "NOT":
            value[s] = 1 - value[args[0]]
        else:
            value[s] = value[args[0]]
    return value[circuit["output"]]


def bench_to_networkx(path, inverters_as_nodes=False):
    """Directed graph in signal direction.

    Default (AIG form): nodes are inputs, AND gates and one output node; NOT gates become
    an edge attribute inverted=True. Node attribute 'type' is 'input' | 'and' | 'output'
    (or 'not' when inverters_as_nodes=True); inputs carry 'position' = their index.
    """
    import networkx as nx
    c = parse_bench(path)
    g = nx.DiGraph()
    for i, x in enumerate(c["inputs"]):
        g.add_node(x, type="input", position=i)
    if inverters_as_nodes:
        for s, (op, args) in c["gates"].items():
            g.add_node(s, type={"AND": "and", "NOT": "not"}.get(op, "buf"))
            for a in args:
                g.add_edge(a, s)
        g.add_node("OUT", type="output")
        g.add_edge(c["output"], "OUT")
        return g
    literal = {x: (x, False) for x in c["inputs"]}
    for s, (op, args) in c["gates"].items():
        if op == "AND":
            g.add_node(s, type="and")
            for a in args:
                node, inv = literal[a]
                g.add_edge(node, s, inverted=inv)
            literal[s] = (s, False)
        else:
            node, inv = literal[args[0]]
            literal[s] = (node, inv ^ (op == "NOT"))
    node, inv = literal[c["output"]]
    g.add_node("OUT", type="output")
    g.add_edge(node, "OUT", inverted=inv)
    return g


# ---------------------------------------------------------------- optional pre-built arrays
def load_graph_arrays(track, root=HERE):
    """Pre-built AIG arrays from graphs/<track>.npz as {sample_id: dict of numpy arrays}.

    node_type: 0 input, 1 AND, 2 output.  pi_index: input position (-1 otherwise).
    level: logic depth of the node.  edge_src/edge_dst: local node ids (signal direction).
    edge_inv: 1 if the edge carries an inverter.
    """
    import numpy as np
    with np.load(os.path.join(root, "graphs", f"{track}.npz")) as z:
        a = {k: z[k] for k in z.files}  # decompress each array once
    out = {}
    for i, sid in enumerate(a["sample_ids"].tolist()):
        n0, n1 = a["node_ptr"][i], a["node_ptr"][i + 1]
        e0, e1 = a["edge_ptr"][i], a["edge_ptr"][i + 1]
        out[sid] = dict(node_type=a["node_type"][n0:n1], pi_index=a["pi_index"][n0:n1], level=a["level"][n0:n1],
                        edge_src=a["edge_src"][e0:e1], edge_dst=a["edge_dst"][e0:e1], edge_inv=a["edge_inv"][e0:e1])
    return out


def to_pyg(arrays, positional=True, max_inputs=256):
    """torch_geometric Data from one array dict.

    x = one-hot node type (3 columns). With positional=True a 4th column holds the input
    position scaled to [0, 1] (0 for non-inputs). Input order matters for the labels, so a
    model that ignores input positions cannot separate functions that differ only by it.
    edge_type: 0 plain wire, 1 inverted wire (for RGCN-style models).
    """
    import torch
    from torch_geometric.data import Data
    nt = torch.as_tensor(arrays["node_type"], dtype=torch.long)
    x = torch.nn.functional.one_hot(nt, 3).float()
    if positional:
        pos = torch.as_tensor(arrays["pi_index"], dtype=torch.float)
        x = torch.cat([x, ((pos + 1).clamp(min=0) / max_inputs).unsqueeze(1)], dim=1)
    edge_index = torch.stack([torch.as_tensor(arrays["edge_src"], dtype=torch.long),
                              torch.as_tensor(arrays["edge_dst"], dtype=torch.long)])
    return Data(x=x, edge_index=edge_index, edge_type=torch.as_tensor(arrays["edge_inv"], dtype=torch.long),
                level=torch.as_tensor(arrays["level"], dtype=torch.long))


def load_pyg(track, split=None, root=HERE, positional=True, include_near_miss=True):
    """List of Data objects with .sample_id, .function_id and integer label .y."""
    import torch
    arrays = load_graph_arrays(track, root)
    rows = read_samples(root, track=track, split=split)
    if not include_near_miss:
        rows = [r for r in rows if r["role"] == "view"]
    label = {f: i for i, f in enumerate(sorted({r["function_id"] for r in rows}))}
    data = []
    for r in rows:
        d = to_pyg(arrays[r["sample_id"]], positional)
        d.y = torch.tensor([label[r["function_id"]]])
        d.sample_id, d.function_id = r["sample_id"], r["function_id"]
        data.append(d)
    return data
