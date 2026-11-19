"""
strict_eval.py -- "strict" grader: swapped-input copies count as WRONG answers.

The official evaluate.py removes swapped-input copies (same perm_group_id, different function)
before scoring. This grader keeps them in, so its full-search scores match exactly what
top20_cec.py shows. Ranking rule (shared with top20_cec.py): highest cosine similarity first,
ties broken by sample_id.

Hard lineups contain no swapped-input copies, so strict hard = official hard.
Only needs numpy.

Example:
    python strict_eval.py --data_root /content/circuitree_v2 --emb /content/expB_real_val.npz
    python strict_eval.py ... --query r029642     # also show one query, to compare with its top-20 table
"""
import argparse, csv, os, sys
import numpy as np

KS = (1, 5, 10, 20)


def add_input_count(E, num_inputs, weight=10.0, max_inputs=256):
    """Input-count rule (same as fgnn2_pos.add_input_count): circuits with a different number of
    inputs can never look close."""
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    oh = np.zeros((len(E), max_inputs + 1))
    oh[np.arange(len(E)), np.minimum(np.asarray(num_inputs), max_inputs)] = weight
    return np.hstack([E, oh])


def rank(sim, ids_arr):
    """Order used by both strict_eval.py and top20_cec.py: highest similarity first, ties by sample_id."""
    return np.lexsort((ids_arr, -sim))


def query_metrics(is_twin_ranked):
    pos = np.flatnonzero(is_twin_ranked)
    if len(pos) == 0:
        return None
    m = {'hit1': float(is_twin_ranked[0]), 'mrr': 1.0 / (pos[0] + 1)}
    for k in KS[1:]:
        m[f'r{k}'] = is_twin_ranked[:k].sum() / len(pos)          # share of all twins in the top k
    m['any20'] = float(pos[0] < 20)                                 # at least one twin in the top 20
    m['ap'] = float((np.arange(1, len(pos) + 1) / (pos + 1)).mean())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--emb', required=True)
    ap.add_argument('--track', default='real')
    ap.add_argument('--split', default='val')
    ap.add_argument('--query', default=None, help='also print this one query (to check against top20_cec.py)')
    ap.add_argument('--input_count', action='store_true', help='apply the input-count rule first')
    a = ap.parse_args()

    sys.path.insert(0, a.data_root)
    import evaluate as ev

    with open(os.path.join(a.data_root, 'samples.csv'), newline='') as f:
        info = {r['sample_id']: r for r in csv.DictReader(f)}
    z = np.load(a.emb)
    ids = [str(s) for s in z['sample_ids']]
    ids_arr = np.array(ids)
    E = z['embeddings'].astype(np.float64)
    if a.input_count:
        E = add_input_count(E, [int(info[s]['num_inputs']) for s in ids])
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    fn = np.array([info[s]['function_id'] for s in ids])
    idx = {s: i for i, s in enumerate(ids)}
    with open(os.path.join(a.data_root, 'eval', f'{a.track}_{a.split}_queries.csv'), newline='') as f:
        queries = [r['query_id'] for r in csv.DictReader(f)]

    res = []
    for q in queries:
        i = idx[q]
        sim = E @ E[i]
        order = rank(sim, ids_arr)
        order = order[order != i]                                   # not itself
        m = query_metrics(fn[order] == fn[i])
        if m:
            res.append(m)
            if q == a.query:
                print(f"query {q}: Hit@1 {m['hit1']:.0f} | MRR {m['mrr']:.3f} | "
                      f"twins in top 10 {m['r10']:.2f} | top 20 {m['r20']:.2f}\n")
    strict = {k: float(np.mean([r[k] for r in res])) for k in res[0]}

    emb = dict(zip(ids, E))
    off_full = ev.evaluate(emb, a.track, a.split, 'full', root=a.data_root)
    off_hard = ev.evaluate(emb, a.track, a.split, 'hard', root=a.data_root)

    print(f"{a.track} {a.split} | {os.path.basename(a.emb)}{' + input-count rule' if a.input_count else ''} | "
          f"{len(res)} full-search queries")
    print(f"{'':28}{'Hit@1':>7}{'MRR':>7}{'top5':>7}{'top10':>7}{'top20':>7}{'>=1 in 20':>11}{'AP':>7}")
    print(f"{'full, STRICT (copies wrong)':28}{strict['hit1']:7.3f}{strict['mrr']:7.3f}{strict['r5']:7.3f}"
          f"{strict['r10']:7.3f}{strict['r20']:7.3f}{strict['any20']:11.3f}{strict['ap']:7.3f}")
    print(f"{'full, official (copies out)':28}{off_full['hit1']:7.3f}{off_full['mrr']:7.3f}{off_full['r5']:7.3f}"
          f"{off_full['r10']:7.3f}{'-':>7}{'-':>11}{off_full['ap']:7.3f}")
    print(f"{'hard (same either way)':28}{off_hard['hit1']:7.3f}{off_hard['mrr']:7.3f}{off_hard['r5']:7.3f}"
          f"{off_hard['r10']:7.3f}{'-':>7}{'-':>11}{off_hard['ap']:7.3f}")
    print('\ntopK = share of ALL of a query\'s twins that land in the top K (averaged over queries)')


if __name__ == '__main__':
    main()
