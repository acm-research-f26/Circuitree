# convert circuits to AIG + load them as graphs (PI / AND / PO nodes, inverted edges)
import os, re, glob, shutil, subprocess
import networkx as nx




def make_aigs(src_dir, out_dir="aig"):
    # run ABC `strash` on every .bench in src_dir and save the AIG to out_dir
    shutil.rmtree(out_dir, ignore_errors=True)      # start clean so old files don't sneak in
    os.makedirs(out_dir)
    paths = sorted(glob.glob(os.path.join(src_dir, "**", "*.bench"), recursive=True))
    for p in paths:
        out = os.path.join(out_dir, os.path.basename(p))
        r = subprocess.run(["yosys-abc", "-c", f'read_bench "{p}"; strash; write_bench -l "{out}"'],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            print("ABC failed on", p, r.stdout[-300:], r.stderr[-300:])
    print(f"{len(paths)} circuits -> {out_dir}/")
    return out_dir




class NotAIGError(ValueError):
    pass


_GATE = re.compile(r"^\s*(\S+)\s*=\s*([A-Za-z_]\w*)\s*(0x[0-9a-fA-F]+)?\s*\((.*)\)\s*$")
_IO = re.compile(r"^\s*(INPUT|OUTPUT)\s*\(\s*([^\s\)]+)\s*\)\s*$", re.I)
_CONST = re.compile(r"^\s*(\S+)\s*=\s*(vdd|gnd|CONST0|CONST1|0|1)\s*(\(\s*\))?\s*$", re.I)


def load_aig(path):
    # read an AIG .bench (ABC `write_bench -l` output) into a networkx DiGraph
    defs, pis, pos = {}, [], []
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            m = _IO.match(line)
            if m:
                (pis if m.group(1).upper() == "INPUT" else pos).append(m.group(2))
                continue
            m = _CONST.match(line)
            if m:
                defs[m.group(1)] = ("CONST", [], "1" if m.group(2).lower() in ("vdd", "const1", "1") else "0")
                continue
            m = _GATE.match(line)
            if m:
                out, op, hexv, args = m.groups()
                defs[out] = (op.upper(), [a.strip() for a in args.split(",") if a.strip()], hexv)

    G = nx.DiGraph()
    for p in pis:
        G.add_node(p, gate="INPUT")
    lit = {p: (p, False) for p in pis}          # signal -> (driving node, inverted?)

    def make_and(name, fanins, invs, out_inv=False):
        G.add_node(name, gate="AND")
        for (node, inv), extra in zip(fanins, invs):
            G.add_edge(node, name, inverted=inv ^ extra)
        return (name, out_inv)

    def build(s):
        op, args, hexv = defs[s]
        ins = [lit[a] for a in args]
        if op == "CONST":
            G.add_node("CONST0", gate="CONST")
            return ("CONST0", hexv == "1")
        if op == "NOT":
            return (ins[0][0], not ins[0][1])
        if op in ("BUF", "BUFF"):
            return ins[0]
        if op == "AND":
            return make_and(s, ins, [False] * len(ins))
        if op == "LUT":                        # write_bench without -l
            tt, k = int(hexv, 16), len(ins)
            if k == 1:
                return (ins[0][0], ins[0][1] ^ bool(tt & 1))
            bits = [(tt >> i) & 1 for i in range(1 << k)]
            for want, out_inv in ((1, False), (0, True)):
                hits = [i for i, b in enumerate(bits) if b == want]
                if len(hits) == 1:             # AND of (possibly inverted) inputs
                    return make_and(s, ins, [not (hits[0] >> j) & 1 for j in range(k)], out_inv)
        raise NotAIGError(f"{os.path.basename(path)}: '{s} = {op}(...)' is not AIG. "
                          "Run make_aigs() (ABC strash) on it first.")

    def resolve(sig):                          # iterative, so deep circuits don't hit recursion limits
        stack = [sig]
        while stack:
            s = stack[-1]
            if s in lit:
                stack.pop(); continue
            if s not in defs:
                raise ValueError(f"{os.path.basename(path)}: signal '{s}' is used but never defined")
            todo = [a for a in defs[s][1] if a not in lit]
            if todo:
                stack.extend(todo); continue
            stack.pop()
            lit[s] = build(s)
        return lit[sig]

    for i, po in enumerate(pos):
        node, inv = resolve(po)
        G.add_node(f"PO{i}:{po}", gate="OUTPUT")
        G.add_edge(node, f"PO{i}:{po}", inverted=inv)
    return G


def load_folder(folder):
    # load every .bench in a folder -> {circuit_name: graph}
    paths = sorted(glob.glob(os.path.join(folder, "**", "*.bench"), recursive=True))
    return {os.path.splitext(os.path.basename(p))[0]: load_aig(p) for p in paths}


def aig_stats(G):
    # [#nodes, #edges, depth, #PI, #PO]
    level = {}
    for n in nx.topological_sort(G):
        preds = list(G.predecessors(n))
        base = max((level[p] for p in preds), default=0)
        level[n] = base + (1 if G.nodes[n]["gate"] == "AND" else 0)
    gates = [d["gate"] for _, d in G.nodes(data=True)]
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
            "depth": max(level.values(), default=0),
            "PI": gates.count("INPUT"), "PO": gates.count("OUTPUT")}
