# Shireen - baselines

`Shireen_Baselines_Week3.ipynb` is the main notebook (open in Colab, Run all).

`baselines/` has the same code split into files so it can be imported:

| file | what's in it |
|---|---|
| `aig_loader.py` | `make_aigs` (ABC strash), `load_aig` (AIG -> graph w/ inverted edges), `aig_stats` |
| `histogram.py` | baseline 1 - histogram |
| `wl_kernel.py` | baseline 2 - WL kernel (grakel) |
| `structural.py` | baseline 3 - `[#nodes, #edges, depth, #PI, #PO]` |
| `evaluate.py` | Hit@1 / P@k / MRR + `embedding_ranker` for the GNN |
| `standin_variants.py` | temp ABC variants until the team ones are ready |

```python
from baselines import *
make_aigs("bench_folder", "aig")
circuits = load_folder("aig")
evaluate(structural_rank, circuits)
```



Results so far (11 ISCAS'85 circuits + temp variants), top-1 match: histogram ~80%, WL ~93%, structural ~94%.
