# FGNN2 evaluation (Jotish)

Evaluating and improving the open-source FGNN2 model (https://github.com/ZeayW/FGNN2) for circuit retrieval
on the Circuitree v2 dataset: retrieval@k, accuracy (Hit@1), MRR, and time.

## Final results (real track, TEST split, official `evaluate.py`)

Final model = **Exp C**: pretrained FGNN2 + fine-tuning (Exp A) + input positions (Exp B) + the input-count rule (Exp C).
Exp C adds no training: it is Exp B's weights plus the rule that circuits with a different number of inputs are never matched.
Chosen on validation before test was run; test run once.

| Model (test) | Hit@1 hard | MRR hard | Hit@1 full |
|---|---|---|---|
| Pretrained FGNN2 | 0.843 | 0.910 | 0.570 |
| + Fine-tuning (Exp A) | 0.872 | 0.927 | 0.640 |
| + Input positions (Exp B) | 0.869 | 0.927 | 0.669 |
| **+ Input-count rule (Exp C, final)** | **0.869** | **0.927** | **0.783** |
| FGNN2 top 20 + CEC | - | - | 0.92 |

Baselines on test: size stats 0.569 / 0.393, WL kernel 0.770 / 0.437, RGCN (20k steps) 0.811 / 0.608,
simulation 0.956 / 0.930 (hard / full Hit@1).

FGNN2 shortlist + CEC on test (swapped-input copies kept): FGNN2 alone 73.1%, top 10 89.8%, top 20 92.3%,
top 50 94.4%, top 100 95.8%. About 0.8 ms per circuit to fingerprint; CEC on the whole library takes 8-20 min per search.

Everything below is the full history (validation numbers unless marked).

## Files

| file | what it is |
|---|---|
| `FGNN2_strict_colab.ipynb` | **start here**: Colab notebook, no DGL needed. Scores pretrained / Exp A / Exp B / Exp C (= Exp B + input-count rule) with the strict and official graders, plus a top-20 + CEC table |
| `FGNN2_circuitree_v2.ipynb` | full Colab notebook used for Experiments A, B, C (needs DGL); section 7 scores Exp C |
| `fgnn2_torch.py` | FGNN2 fingerprints in plain PyTorch (same math as FGNN2's FunctionConv, loads the same weights). Made because DGL's download site went down |
| `strict_eval.py` | strict grader: swapped-input copies count as wrong; matches the top-20 tables exactly |
| `top20_cec.py` | top 20 most similar circuits for one query, each checked with ABC `cec` |
| `embed_circuitree.py` | FGNN2 fingerprints with DGL (used for Experiments A-C) |
| `finetune_fgnn2.py` | fine-tuning on the Circuitree v2 train split (Exp A; `--positions` = Exp B) |
| `fgnn2_pos.py` | builds FGNN2 with input positions; the input-count rule (Exp C) |
| `eval_retrieval.py` | `.bench` parser + first ISCAS'85 twin test |
| `fast_nce.py`, `PATCHES.md` | faster training loss; bugs found in the FGNN2 repo |
| `expA_*`, `expB_*` | training logs (`*_log.csv`) and fine-tuned weights (`*_best.pth`). **Exp C = Exp B weights + the input-count rule**, so it has no separate weights |
| `top20_*.csv` | top-20 + CEC tables for query r029642 |
| `poster_images/` | figures for the poster |

## Setup (Colab)

DGL has no build for Python 3.13 (Colab's default), so FGNN2 runs inside a Python 3.12
virtual environment with PyTorch 2.4 + DGL 2.4 (CUDA 12.4):

```
!git clone -q https://github.com/ZeayW/FGNN2.git
!pip install -q uv
!uv venv /content/fgnn_env --python 3.12 --seed
!/content/fgnn_env/bin/pip install -q "numpy<2" torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
!/content/fgnn_env/bin/pip install -q packaging pydantic pyyaml pandas psutil tqdm requests networkx scipy
!/content/fgnn_env/bin/pip install -q "dgl==2.4.0+cu124" -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
!/content/fgnn_env/bin/pip install -q "torchdata==0.8.0" --no-deps
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
So we moved to the harder Circuitree v2 dataset (below), which has near-miss imposters.

## Circuitree v2 results (real track, val split, official `evaluate.py`)

| | Hit@1 hard | MRR hard | Hit@1 full |
|---|---|---|---|
| wl_kernel (dataset baseline) | 0.656 | | 0.373 |
| untrained FGNN2 | 0.453 | 0.638 | 0.021 |
| pretrained FGNN2 | 0.780 | 0.869 | 0.616 |
| Exp A: fine-tuned 2000 steps on train | 0.819 | 0.893 | 0.660 |
| Exp B: + input positions | 0.821 | 0.896 | 0.676 |
| Exp C: + input-count rule | 0.821 | 0.896 | 0.738 |

~0.8 ms per circuit to embed on a Colab T4. Val scores move about +-1 point between checkpoints, so A vs B on
hard is a tie. The grader skips swapped-input (same perm group) circuits; in the top-20 table, pretrained and
Exp A give swapped-input copies the exact same score as real twins (1.0000), while Exp B separates them
(twin 0.9993, copies 0.983 / 0.973). Test split not run yet (run once at the end).

## Strict scores (swapped-input copies count as wrong), full search, val, input-count rule

| Model | Hit@1 strict | Hit@1 official | twins in top 10 | twins in top 20 | >=1 twin in top 20 |
|---|---|---|---|---|---|
| pretrained | 0.593 | 0.688 | 0.524 | 0.611 | 0.820 |
| Exp A | 0.628 | 0.723 | 0.576 | 0.667 | 0.829 |
| Exp C (Exp B + rule) | 0.691 | 0.738 | 0.603 | 0.698 | 0.910 |

Officially A and B tie, but under the strict rule input positions add +6.3 Hit@1 (Exp A 0.628 -> Exp C 0.691).
