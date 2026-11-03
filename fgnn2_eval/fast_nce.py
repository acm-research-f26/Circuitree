"""
fast_nce.py -- drop-in, vectorized replacements for the NCE losses in pretrain.py.

The original code does `for j in range(num_pair): loss += Loss(embeddings, codes, j, j+num_pair, tao)`
which launches ~6 small GPU ops per pair (thousands per batch). These compute the same
quantity for the whole batch at once.

In pretrain.py replace:
    for j in range(num_pair):
        loss += Loss(embeddings,codes,j,j+num_pair,options.tao)
    loss = loss / (2*num_pair)
with:
    from fast_nce import nce_loss_fast
    loss = nce_loss_fast(embeddings, codes, options.tao, weighted=options.weighted)
"""
import torch as th
import torch.nn.functional as F


def nce_loss_fast(embeddings, codes, tao, weighted=True):
    """embeddings: [2n, d]  (first n = view 1, last n = view 2 of the same circuits)
    codes:      [2n, c]  truth-table codes (duplicated, as in pretrain.py)
    weighted=True  -> same as NCEloss_w (negatives down-weighted by 1 - code similarity)
    weighted=False -> same as NCEloss, except it removes the true self-term
                      instead of subtracting exp(1/tao)."""
    two_n = embeddings.shape[0]; n = two_n // 2
    z = F.normalize(embeddings, dim=-1)
    S = z @ z.T / tao                                        # [2n, 2n]
    idx = th.arange(two_n, device=z.device)
    partner = (idx + n) % two_n
    pos = S[idx, partner]                                    # [2n]

    if weighted:
        c = F.normalize(codes.float(), dim=-1)
        W = (1 - c @ c.T).clamp(min=0)                       # 0 for same-function pairs (and self)
        neg = th.logsumexp(S + th.log(W + 1e-30), dim=1)     # log sum_k W*exp(S)
        denom = th.logaddexp(neg, pos)                       # + exp(pos)
    else:
        S_noself = S.masked_fill(th.eye(two_n, dtype=th.bool, device=z.device), float('-inf'))
        denom = th.logsumexp(S_noself, dim=1)

    return (denom - pos).mean()                              # == sum over both directions / (2n)


if __name__ == '__main__':          # check against the original loop version
    import math
    def NCEloss_w(e, c, i, j, t):
        pos = th.cosine_similarity(e[i], e[j], dim=-1)
        n1 = th.cosine_similarity(e[i], e, dim=-1); n2 = th.cosine_similarity(e[j], e, dim=-1)
        cs = th.cosine_similarity(c[i], c, dim=-1)
        l12 = -th.log(th.exp(pos/t) / (th.sum((1-cs)*th.exp(n1/t)) + th.exp(pos/t)))
        l21 = -th.log(th.exp(pos/t) / (th.sum((1-cs)*th.exp(n2/t)) + th.exp(pos/t)))
        return l12 + l21
    th.manual_seed(0); n = 16
    e = th.randn(2*n, 32); c = th.rand(n, 6); c = th.cat((c, c))
    ref = sum(NCEloss_w(e, c, j, j+n, 0.1) for j in range(n)) / (2*n)
    print('loop:', ref.item(), ' fast:', nce_loss_fast(e, c, 0.1).item())
