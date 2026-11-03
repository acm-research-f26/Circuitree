"""Score circuit embeddings on the fixed Circuitree retrieval benchmark.

Input: one vector per sample_id, however you produced it.
  .npz with arrays 'sample_ids' (str) and 'embeddings' (N x D), or
  .csv whose first column is sample_id and remaining columns are the vector.

    python evaluate.py --embeddings my_test_embeddings.npz --track real --split test

Conditions
  full : each query is ranked against every other sample of the same track and split.
  hard : each query is ranked against its fixed gallery in eval/<track>_<split>_hard_gallery.csv
         (other views of its function, size-matched different functions with the same number
         of inputs, and one-gate near-miss mutants). Only queries flagged in_hard_eval are used.
In both, functions in the query's permutation group (possibly the same function with inputs
re-ordered) are left out of the ranking: they are neither right nor wrong answers.

Subsets (real track): by default all queries are scored. subset="combinational" keeps only queries
whose function was found in at least one purely combinational chip; subset="sequential" keeps the
rest (functions found only in the logic between flip-flops of sequential chips). Galleries are the
same in every subset.

Similarity is cosine. Ties are broken randomly with a fixed seed.
Metrics: Hit@1, MRR (rank of the first correct answer), Recall@5/10 (share of correct answers
in the top k), mAP.
"""
import argparse
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_embeddings(path):
    if path.endswith(".npz"):
        z = np.load(path)
        return dict(zip(z["sample_ids"].tolist(), np.asarray(z["embeddings"], dtype=np.float64)))
    out = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if row and not row[0].startswith("sample_id"):
                out[row[0]] = np.array([float(v) for v in row[1:]])
    return out


def _rows(name, root):
    with open(os.path.join(root, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _metrics(scores, labels, rng):
    order = np.lexsort((rng.random(len(scores)), -scores))
    hits = np.asarray(labels)[order]
    pos = np.flatnonzero(hits)
    if len(pos) == 0:
        return None
    precision_at_hits = np.arange(1, len(pos) + 1) / (pos + 1)
    return dict(hit1=float(hits[0]), mrr=1.0 / (pos[0] + 1), r5=hits[:5].sum() / len(pos),
                r10=hits[:10].sum() / len(pos), ap=float(precision_at_hits.mean()))


def evaluate(embeddings, track, split="test", condition="hard", root=HERE, seed=0, subset="all"):
    samples = {r["sample_id"]: r for r in _rows("samples.csv", root) if r["track"] == track and r["split"] == split}
    queries = _rows(os.path.join("eval", f"{track}_{split}_queries.csv"), root)
    if subset != "all":
        want = "0" if subset == "combinational" else "1"
        queries = [q for q in queries if samples[q["query_id"]]["from_sequential_design"] == want]
    if condition == "hard":
        queries = [q for q in queries if q["in_hard_eval"] == "1"]
        gallery = {}
        for g in _rows(os.path.join("eval", f"{track}_{split}_hard_gallery.csv"), root):
            gallery.setdefault(g["query_id"], []).append(g["candidate_id"])
    elif condition != "full":
        raise ValueError("condition must be 'hard' or 'full'")
    missing = [s for s in samples if s not in embeddings]
    if missing:
        raise KeyError(f"{len(missing)} samples have no embedding, e.g. {missing[:3]}")

    ids = list(samples)
    index = {s: i for i, s in enumerate(ids)}
    matrix = np.stack([np.asarray(embeddings[s], dtype=np.float32) for s in ids])
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    # Integer codes so candidate filtering is vectorised.
    fn_code = {f: i for i, f in enumerate(sorted({samples[s]["function_id"] for s in ids}))}
    pg_code = {g: i for i, g in enumerate(sorted({samples[s]["perm_group_id"] for s in ids} - {""}))}
    fn = np.array([fn_code[samples[s]["function_id"]] for s in ids])
    pg = np.array([pg_code.get(samples[s]["perm_group_id"], -1) for s in ids])
    q_rows = np.array([index[q["query_id"]] for q in queries], dtype=int)
    sims = matrix[q_rows] @ matrix.T if len(q_rows) else np.zeros((0, len(ids)))  # all query scores at once
    rng = np.random.default_rng(seed)
    results = []
    for qi, q in enumerate(queries):
        r = q_rows[qi]
        if condition == "hard":
            cand = np.array([index[c] for c in gallery[q["query_id"]]], dtype=int)
        else:
            cand = np.delete(np.arange(len(ids)), r)
        same_fn = fn[cand] == fn[r]
        keep = same_fn | (pg[cand] == -1) | (pg[cand] != pg[r])
        m = _metrics(sims[qi, cand[keep]], same_fn[keep], rng)
        if m:
            results.append(m)
    summary = {k: float(np.mean([r[k] for r in results])) if results else float("nan")
               for k in ["hit1", "mrr", "r5", "r10", "ap"]}
    summary.update(queries=len(results), track=track, split=split, condition=condition, subset=subset)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--track", choices=["real", "synthetic"], required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--condition", choices=["hard", "full", "both"], default="both")
    p.add_argument("--subset", choices=["all", "combinational", "sequential"], default="all")
    p.add_argument("--root", default=HERE)
    args = p.parse_args()
    emb = load_embeddings(args.embeddings)
    for c in ["hard", "full"] if args.condition == "both" else [args.condition]:
        print(json.dumps(evaluate(emb, args.track, args.split, c, args.root, subset=args.subset)))


if __name__ == "__main__":
    main()
