"""
embed_circuitree.py -- make FGNN2 fingerprints for the Circuitree v2 dataset.

Put this file in FGNN2/src/ next to eval_retrieval.py (it reuses its parser and batching).
It writes an .npz with 'sample_ids' and 'embeddings', which the dataset's own
evaluate.py scores.

Example:
    python embed_circuitree.py --data_root /content/circuitree_v2 --track real --split val \
        --weights ../checkpoints/example/weight.pth --flag_inv --out /content/fgnn2_real_val.npz
"""
import argparse, csv, os, time
import numpy as np
from eval_retrieval import parse_bench, build_graph, embed_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True, help='the circuitree_v2 folder')
    ap.add_argument('--track', choices=['real', 'synthetic'], required=True)
    ap.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    ap.add_argument('--weights', default=None, help='leave out = untrained (random) FGNN2')
    ap.add_argument('--hidden_dim', type=int, default=256)
    ap.add_argument('--flag_inv', action='store_true')
    ap.add_argument('--flag_proj', action='store_true')
    ap.add_argument('--positions', action='store_true', help='model has input positions (Experiment B/C weights)')
    ap.add_argument('--input_count', action='store_true', help='apply the input-count rule to the fingerprints')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    import torch as th
    from fgnn2_pos import make_model, add_input_count
    device = th.device(f'cuda:{a.gpu}' if th.cuda.is_available() else 'cpu')
    sync = (lambda: th.cuda.synchronize()) if device.type == 'cuda' else (lambda: None)

    model = make_model(a.hidden_dim, positions=a.positions, weights=a.weights, device=device)
    model.set_flag_proj(a.flag_proj); model.flag_inv = a.flag_inv
    if not a.weights:
        print('WARNING: no --weights, using an untrained (random) FGNN2')
    model.eval()

    # every circuit of this track + split (near-miss samples included, evaluate.py needs them)
    with open(os.path.join(a.data_root, 'samples.csv'), newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['track'] == a.track and r['split'] == a.split]
    # circuits of similar depth go in the same batch -> fewer wasted steps
    rows.sort(key=lambda r: int(r['depth']))
    print(f'{len(rows)} circuits ({a.track} {a.split})')

    t0 = time.perf_counter()
    parsed = [parse_bench(os.path.join(a.data_root, r['file'])) for r in rows]
    graphs = [build_graph(p) for p in parsed]
    t_load = time.perf_counter() - t0

    embs = []
    with th.no_grad():
        embed_batch(model, graphs[:2], [p['po_nids'] for p in parsed[:2]],
                    [p['po_inv'] for p in parsed[:2]], device)            # warm-up
        sync(); t0 = time.perf_counter()
        for s in range(0, len(graphs), a.batch_size):
            e = embed_batch(model, graphs[s:s + a.batch_size],
                            [p['po_nids'] for p in parsed[s:s + a.batch_size]],
                            [p['po_inv'] for p in parsed[s:s + a.batch_size]], device)
            embs.append(e.cpu().numpy())
            if (s // a.batch_size) % 10 == 0:
                print(f'  {min(s + a.batch_size, len(graphs))}/{len(graphs)} done')
        sync(); t_embed = time.perf_counter() - t0

    E = np.concatenate(embs)
    if a.input_count:
        E = add_input_count(E, [p['n_pi'] for p in parsed])
        print('input-count rule applied')
    np.savez(a.out, sample_ids=np.array([r['sample_id'] for r in rows]), embeddings=E)
    print(f'saved {a.out}')
    print(f'time to read files: {t_load:.1f} s')
    print(f'time to embed:      {t_embed:.1f} s  ({1000 * t_embed / len(rows):.2f} ms per circuit)')


if __name__ == '__main__':
    main()
