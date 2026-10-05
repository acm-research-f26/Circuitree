# temporary variants until the team ones are done
import os, glob, subprocess
import warnings
import networkx as nx
from .aig_loader import load_aig





VARIANT_SCRIPTS = {
    "v1": "balance", "v2": "rewrite", "v3": "refactor", "v4": "balance; rewrite; refactor",
    "v5": "rewrite -z; refactor -z",
    "v6": "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance",
    "v7": "dc2", "v8": "if -K 4; strash", "v9": "if -K 6; strash; balance", "v10": "drw; drf",
    "v11": "if -K 3; strash; rewrite", "v12": "resub; rewrite -z",
}


def make_standin_variants(orig_dir, out_dir="standin_variants"):
    # makes up to 12 variants per circuit, CEC checks them, drops duplicates
    os.makedirs(out_dir, exist_ok=True)
    abc = lambda cmd: subprocess.run(["yosys-abc", "-c", cmd], capture_output=True, text=True).stdout

    def body(f):   # structure fingerprint: same graph = same hash, whatever the wire names are
        G = load_aig(f)
        for u, v, d in G.edges(data=True):
            d["inv"] = str(d["inverted"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return nx.weisfeiler_lehman_graph_hash(G, node_attr="gate", edge_attr="inv", iterations=5)
    for p in sorted(glob.glob(os.path.join(orig_dir, "*.bench"))):
        c = os.path.splitext(os.path.basename(p))[0]
        base = os.path.join(out_dir, f"_{c}_strash.bench")
        abc(f'read_bench "{p}"; strash; write_bench -l "{base}"')
        seen, kept = {body(base)}, 0
        for v, script in VARIANT_SCRIPTS.items():
            out = os.path.join(out_dir, f"{c}_{v}.bench")
            abc(f'read_bench "{p}"; strash; {script}; strash; write_bench -l "{out}"')
            ok = "are equivalent" in abc(f'cec "{p}" "{out}"')
            h = body(out)
            if not ok or h in seen:
                os.remove(out)
                if not ok: print("  CEC failed, dropped:", out)
                continue
            seen.add(h); kept += 1
        os.remove(base)
        print(f"{c}: {kept} variants")
    return out_dir
