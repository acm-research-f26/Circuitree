"""
finetune_fgnn2.py -- fine-tune FGNN2 on the Circuitree v2 train split.

Experiment A: same model, only the weights get more practice.
Experiment B (--positions): each input also gets its own learned starting fingerprint.
Experiment C: no training; Exp B's weights + the input-count rule (fgnn2_pos.add_input_count).
Put this file in FGNN2/src/ next to eval_retrieval.py.

How one training step works ("pull and push"):
  * pick a group of different functions (families) with similar depth
  * take 2 twins (views) from each family            -> PULL their fingerprints together
  * every other circuit in the step is a negative     -> PUSH them apart
  * near-miss imposters of those families are extra negatives
Every --eval_every steps it scores the val split with the dataset's own evaluate.py
and saves the best weights (by val hard Hit@1). The test split is never touched.

Example:
    python finetune_fgnn2.py --data_root /content/circuitree_v2 \
        --weights ../checkpoints/example/weight.pth --steps 2000 --out_dir /content/expA
"""
import argparse, csv, os, random, sys, time
import numpy as np
from eval_retrieval import parse_bench, build_graph, embed_batch


def load_split(root, track, split):
    with open(os.path.join(root, 'samples.csv'), newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['track'] == track and r['split'] == split]
    t = time.perf_counter()
    parsed = [parse_bench(os.path.join(root, r['file'])) for r in rows]
    graphs = [build_graph(p) for p in parsed]
    print(f'  loaded {len(rows)} {split} circuits in {time.perf_counter() - t:.0f} s')
    return rows, parsed, graphs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--track', default='real')
    ap.add_argument('--weights', default=None, help='start from these weights (leave out = from scratch)')
    ap.add_argument('--hidden_dim', type=int, default=256)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch_functions', type=int, default=64, help='families per step')
    ap.add_argument('--node_budget', type=int, default=60000, help='max AND gates per step (avoids running out of GPU memory)')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--temperature', type=float, default=0.1)
    ap.add_argument('--no_near_miss', action='store_true', help="don't use imposters as extra negatives")
    ap.add_argument('--depth_bins', type=int, default=8)
    ap.add_argument('--eval_every', type=int, default=250)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--positions', action='store_true', help='Experiment B: give each input its own position')
    ap.add_argument('--pos_lr', type=float, default=1e-3, help='learning rate for the new position vectors')
    ap.add_argument('--select', choices=['hard', 'full_rule'], default='hard',
                    help='which val score picks the best weights (full_rule = full Hit@1 with the input-count rule)')
    ap.add_argument('--out_dir', required=True)
    a = ap.parse_args()

    import torch as th
    import torch.nn.functional as F
    from fgnn2_pos import make_model, add_input_count
    sys.path.insert(0, a.data_root)
    import evaluate as ev

    random.seed(a.seed); np.random.seed(a.seed); th.manual_seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    model = make_model(a.hidden_dim, positions=a.positions, weights=a.weights, device=device)
    print('starting from', 'weights ' + a.weights if a.weights else 'random weights (from scratch)',
          '| input positions:', 'ON' if a.positions else 'off')

    print('loading data (takes a minute or two)...')
    tr_rows, tr_parsed, tr_graphs = load_split(a.data_root, a.track, 'train')
    va_rows, va_parsed, va_graphs = load_split(a.data_root, a.track, 'val')
    va_order = sorted(range(len(va_rows)), key=lambda i: int(va_rows[i]['depth']))

    # ---- group the training circuits into families
    by_fn, mutants_of, pg_of, depth_of, size_of = {}, {}, {}, {}, {}
    for i, r in enumerate(tr_rows):
        if r['role'] == 'view':
            by_fn.setdefault(r['function_id'], []).append(i)
            pg_of[r['function_id']] = r['perm_group_id']
        else:
            mutants_of.setdefault(r['source_function_id'], []).append(i)
    functions = [f for f, v in by_fn.items() if len(v) >= 2]
    for f in functions:
        depth_of[f] = float(np.median([int(tr_rows[i]['depth']) for i in by_fn[f]]))
        size_of[f] = float(np.mean([int(tr_rows[i]['and_count']) for i in by_fn[f]]))
    # depth bins: each step uses families of similar depth (faster + harder negatives)
    edges = np.quantile([depth_of[f] for f in functions], np.linspace(0, 1, a.depth_bins + 1))
    bins = [[] for _ in range(a.depth_bins)]
    for f in functions:
        b = min(int(np.searchsorted(edges, depth_of[f], side='right')) - 1, a.depth_bins - 1)
        bins[max(b, 0)].append(f)
    bins = [b for b in bins if len(b) >= 8]
    print(f'train: {len(functions)} families, {sum(len(v) for v in mutants_of.values())} imposters, '
          f'{len(bins)} depth bins')

    def batch_embed(idx_list, graphs, parsed):
        return embed_batch(model, [graphs[i] for i in idx_list], [parsed[i]['po_nids'] for i in idx_list],
                           [parsed[i]['po_inv'] for i in idx_list], device)

    def score_val():
        model.eval()
        embs = {}
        with th.no_grad():
            for s in range(0, len(va_order), 128):
                chunk = va_order[s:s + 128]
                e = batch_embed(chunk, va_graphs, va_parsed).cpu().numpy()
                for i, v in zip(chunk, e):
                    embs[va_rows[i]['sample_id']] = v
        model.train()
        hard = ev.evaluate(embs, a.track, 'val', 'hard', root=a.data_root)
        full = ev.evaluate(embs, a.track, 'val', 'full', root=a.data_root)
        ids = list(embs)
        ruled = add_input_count(np.stack([embs[s] for s in ids]), [n_in[s] for s in ids])
        full_rule = ev.evaluate(dict(zip(ids, ruled)), a.track, 'val', 'full', root=a.data_root)
        return hard, full, full_rule

    n_in = {va_rows[i]['sample_id']: va_parsed[i]['n_pi'] for i in range(len(va_rows))}
    pick = lambda hard, full_rule: hard['hit1'] if a.select == 'hard' else full_rule['hit1']

    def info_nce(za, zb, extra):
        logits = za @ zb.t()
        if extra is not None:
            logits = th.cat([logits, za @ extra.t()], 1)
        return F.cross_entropy(logits / a.temperature, th.arange(len(za), device=za.device))

    pos_params = list(model.pos_emb.parameters()) if a.positions else []
    other = [p for p in model.parameters() if all(p is not q for q in pos_params)]
    groups = [{'params': other, 'lr': a.lr}]
    if pos_params:
        groups.append({'params': pos_params, 'lr': a.pos_lr})   # brand-new, so it learns faster
    opt = th.optim.Adam(groups)
    log_path = os.path.join(a.out_dir, 'log.csv')
    log = open(log_path, 'w', newline='')
    logw = csv.writer(log)
    logw.writerow(['step', 'avg_loss', 'val_hard_hit1', 'val_hard_mrr', 'val_full_hit1',
                   'val_full_mrr', 'val_full_rule_hit1', 'val_full_rule_mrr', 'minutes'])

    hard, full, full_rule = score_val()
    best = pick(hard, full_rule)
    print(f"step 0 (before training): val hard Hit@1 {hard['hit1']:.4f} | full Hit@1 {full['hit1']:.4f} "
          f"| full + input-count rule {full_rule['hit1']:.4f}")
    logw.writerow([0, '', hard['hit1'], hard['mrr'], full['hit1'], full['mrr'],
                   full_rule['hit1'], full_rule['mrr'], 0]); log.flush()
    th.save(model.state_dict(), os.path.join(a.out_dir, 'best.pth'))

    t_start = time.perf_counter()
    losses = []
    model.train()
    for step in range(1, a.steps + 1):
        fams = random.choice(bins)[:]
        random.shuffle(fams)
        avg_size = np.mean([size_of[f] for f in fams])
        B = int(max(8, min(a.batch_functions, a.node_budget // (3 * max(avg_size, 1)))))
        chosen, used_pg = [], set()
        for f in fams:
            # without positions, swapped-input families look identical to FGNN2, so keep only one
            # per step. With positions the model CAN tell them apart, so they stay as hard negatives.
            if not a.positions and pg_of[f] and pg_of[f] in used_pg:
                continue
            chosen.append(f); used_pg.add(pg_of[f])
            if len(chosen) == B:
                break
        pairs = [random.sample(by_fn[f], 2) for f in chosen]
        negs = [] if a.no_near_miss else [random.choice(mutants_of[f]) for f in chosen if f in mutants_of]

        idx = [p[0] for p in pairs] + [p[1] for p in pairs] + negs     # one big batch = one pass
        z = F.normalize(batch_embed(idx, tr_graphs, tr_parsed), dim=1)
        n = len(pairs)
        za, zb, zn = z[:n], z[n:2 * n], (z[2 * n:] if negs else None)
        loss = (info_nce(za, zb, zn) + info_nce(zb, za, zn)) / 2

        opt.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if step % 50 == 0:
            mins = (time.perf_counter() - t_start) / 60
            print(f'step {step}: loss {np.mean(losses[-50:]):.3f}  ({mins:.1f} min)')
        if step % a.eval_every == 0 or step == a.steps:
            hard, full, full_rule = score_val()
            mins = (time.perf_counter() - t_start) / 60
            tag = ''
            if pick(hard, full_rule) > best:
                best = pick(hard, full_rule)
                th.save(model.state_dict(), os.path.join(a.out_dir, 'best.pth'))
                tag = '  <- new best, saved'
            print(f"  VAL step {step}: hard Hit@1 {hard['hit1']:.4f} | full Hit@1 {full['hit1']:.4f} | "
                  f"full + rule {full_rule['hit1']:.4f}{tag}")
            logw.writerow([step, np.mean(losses[-a.eval_every:]),
                           hard['hit1'], hard['mrr'], full['hit1'], full['mrr'],
                           full_rule['hit1'], full_rule['mrr'], round(mins, 2)]); log.flush()

    th.save(model.state_dict(), os.path.join(a.out_dir, 'last.pth'))
    print(f"\ndone. best val {'hard Hit@1' if a.select == 'hard' else 'full Hit@1 with input-count rule'} = {best:.4f}")
    print(f"weights: {os.path.join(a.out_dir, 'best.pth')}   log: {log_path}")


if __name__ == '__main__':
    main()
