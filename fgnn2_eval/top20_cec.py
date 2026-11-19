"""
top20_cec.py -- for one query circuit, show the top-k most similar circuits by FGNN2
fingerprint, and check each one with ABC's CEC (combinational equivalence checking).

Needs: the .npz from embed_circuitree.py, the circuitree_v2 folder, and yosys-abc
(Colab: !apt-get -qq install -y yosys). Only numpy is needed, no PyTorch.

Example:
    python top20_cec.py --data_root /content/circuitree_v2 --emb /content/fgnn2_real_val.npz \
        --track real --split val --query r027344 --k 20 --out /content/top20_r027344.csv
Leave out --query to pick a random evaluation query (use --seed to change which one).
"""
import argparse, csv, os, random, subprocess, time
import numpy as np


def add_input_count(E, num_inputs, weight=10.0, max_inputs=256):
    """Input-count rule (same as fgnn2_pos.add_input_count): circuits with a different number of
    inputs can never look close."""
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    oh = np.zeros((len(E), max_inputs + 1))
    oh[np.arange(len(E)), np.minimum(np.asarray(num_inputs), max_inputs)] = weight
    return np.hstack([E, oh])


def run_cec(abc, a, b):
    """Return (result, seconds). result: EQUIVALENT / NOT EQUIVALENT / DIFFERENT #INPUTS / ERROR"""
    t = time.perf_counter()
    out = subprocess.run([abc, '-q', f'cec {a} {b}'], capture_output=True, text=True, timeout=120)
    sec = time.perf_counter() - t
    text = out.stdout + out.stderr
    if 'NOT EQUIVALENT' in text:
        return 'NOT EQUIVALENT', sec
    if 'are equivalent' in text:
        return 'EQUIVALENT', sec
    if 'different number of primary inputs' in text:
        return 'DIFFERENT #INPUTS', sec
    return 'ERROR', sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--emb', required=True, help='.npz from embed_circuitree.py')
    ap.add_argument('--track', default='real')
    ap.add_argument('--split', default='val')
    ap.add_argument('--query', default=None, help='sample_id; leave out for a random query')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--abc', default='yosys-abc')
    ap.add_argument('--out', default=None, help='optional .csv to save the table')
    ap.add_argument('--input_count', action='store_true', help='apply the input-count rule first')
    a = ap.parse_args()

    with open(os.path.join(a.data_root, 'samples.csv'), newline='') as f:
        info = {r['sample_id']: r for r in csv.DictReader(f)}

    z = np.load(a.emb)
    ids = z['sample_ids'].tolist()
    emb = z['embeddings'].astype(np.float64)
    if a.input_count:
        emb = add_input_count(emb, [int(info[s]['num_inputs']) for s in ids])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12

    q = a.query
    if q is None:
        qfile = os.path.join(a.data_root, 'eval', f'{a.track}_{a.split}_queries.csv')
        with open(qfile, newline='') as f:
            q = random.Random(a.seed).choice([r['query_id'] for r in csv.DictReader(f)])
    qi = ids.index(q)
    qrow = info[q]

    sim = emb @ emb[qi]
    # same ranking rule as strict_eval.py: highest similarity first, ties broken by sample_id
    order = np.lexsort((np.array(ids), -sim))
    top = order[order != qi][:a.k]                      # don't return the query itself

    total_twins = sum(1 for s in ids if s != q and info[s]['function_id'] == qrow['function_id'])
    qpath = os.path.join(a.data_root, qrow['file'])
    print(f"QUERY {q}: {qrow['num_inputs']} inputs, {qrow['and_count']} ANDs, depth {qrow['depth']}, "
          f"from {qrow['source_circuit']}")
    print(f"gallery = all {len(ids) - 1} other circuits in {a.track} {a.split}; "
          f"true twins in gallery: {total_twins}\n")

    header = ['rank', 'sample_id', 'similarity', 'inputs', 'ANDs', 'depth', 'role', 'label_twin', 'CEC', 'cec_s']
    print(f"{'rank':>4}  {'sample_id':<9} {'sim':>7} {'in':>4} {'ANDs':>5} {'dep':>4}  {'role':<9} "
          f"{'twin?':<5}  CEC")
    table, n_eq, agree, t_cec = [], 0, 0, 0.0
    for rank, j in enumerate(top, 1):
        s = ids[j]; r = info[s]
        label = 'yes' if r['function_id'] == qrow['function_id'] else 'no'
        cec, sec = run_cec(a.abc, qpath, os.path.join(a.data_root, r['file']))
        t_cec += sec
        n_eq += cec == 'EQUIVALENT'
        agree += (cec == 'EQUIVALENT') == (label == 'yes')
        table.append([rank, s, f'{sim[j]:.4f}', r['num_inputs'], r['and_count'], r['depth'],
                      r['role'], label, cec, f'{sec:.3f}'])
        print(f"{rank:>4}  {s:<9} {sim[j]:>7.4f} {r['num_inputs']:>4} {r['and_count']:>5} {r['depth']:>4}  "
              f"{r['role']:<9} {label:<5}  {cec}")

    twin_ranks = [row[0] for row in table if row[7] == 'yes']
    strict_mrr = f'{1 / twin_ranks[0]:.3f}' if twin_ranks else f'< {1 / (a.k + 1):.3f}'
    print(f"\nthis query, strict scores (copies count as wrong): Hit@1 {int(bool(twin_ranks) and twin_ranks[0] == 1)} | "
          f"MRR {strict_mrr} | twins in top {a.k}: {len(twin_ranks)}/{total_twins}")
    print(f"CEC says {n_eq} of the top {a.k} are truly equivalent "
          f"(out of {total_twins} twins that exist).")
    print(f"CEC agrees with the dataset labels on {agree}/{a.k}.  Total CEC time: {t_cec:.2f} s")
    if a.out:
        with open(a.out, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(header); w.writerows(table)
        print(f'saved {a.out}')


if __name__ == '__main__':
    main()
