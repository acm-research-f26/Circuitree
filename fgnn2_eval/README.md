# FGNN2 evaluation (Jotish)

Evaluating the open-source FGNN2 model (https://github.com/ZeayW/FGNN2) on circuit
retrieval: retrieval@k, accuracy, MRR, and time.

## Files

| file | what it is |
|---|---|
| `eval_retrieval.py` | the "twin game": embeds every circuit with FGNN2, ranks all others by cosine similarity, reports hit@k, MRR, top-1 accuracy, and timing. Reads ABC AIG `.bench` files (AND / NOT / BUFF / vdd), multi-output. |
| `fast_nce.py` | vectorized version of FGNN2's contrastive (NCE) training loss, for faster pretraining |
| `PATCHES.md` | bugs found in the FGNN2 repo and how to fix them |

## Setup (Colab)

DGL has no build for Python 3.13 (Colab's default), so FGNN2 runs inside a Python 3.12
virtual environment with PyTorch 2.4 + DGL 2.4 (CUDA 12.4):

```
!git clone -q https://github.com/ZeayW/FGNN2.git
!pip install -q uv
!uv venv /content/fgnn_env --python 3.12 --seed
!/content/fgnn_env/bin/pip install -q "numpy<2" torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
!/content/fgnn_env/bin/pip install -q dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
!mkdir -p FGNN2/checkpoints/example
!gdown -q 1NGAJRedx040A4EeN9mALXsly87TRseRF -O FGNN2/checkpoints/example/weight.pth
```

Copy `eval_retrieval.py` into `FGNN2/src/`. Put the variants in one folder per circuit
(`iscas_variants/c432/c432_v1.bench`, ...), then:

```
!cd /content/FGNN2/src && /content/fgnn_env/bin/python eval_retrieval.py \
    --bench_root /content/iscas_variants \
    --weights ../checkpoints/example/weight.pth --flag_inv --ks 1 5 10 --show_misses
```

Leave out `--weights` to test an untrained (random) FGNN2 as a baseline.

## Results so far (Nov 2, 2026)

Test set: 10 ISCAS'85 circuits (c17 skipped) x 10 ABC variants = 100 circuits.
Every circuit is a query; the other 99 are ranked. Twins = same source circuit. GPU: Colab T4.

| | pretrained FGNN2 | untrained FGNN2 |
|---|---|---|
| hit@1 (= top-1 accuracy) | 0.96 | 0.92 |
| hit@5 | 0.99 | 0.99 |
| hit@10 | 1.00 | 1.00 |
| MRR | 0.974 | 0.952 |
| time per circuit (embedding) | ~6 ms | ~6 ms |

All 4 misses of the pretrained model are c1355 <-> c499, which are functionally
equivalent circuits (c1355 = c499 with XORs expanded). Counting those as matches,
pretrained FGNN2 gets 100% top-1.

**Takeaway:** an untrained FGNN2 already scores 92%, so this test mostly measures
circuit *shape*, not *function*. It is too easy to show improvements.
**Next:** build a harder test with "imposter" circuits (tiny edits that change the
function, verified by simulation).
