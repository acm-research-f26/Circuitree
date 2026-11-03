# baselines on dataset v2 (build night 6)

histogram + WL on circuitree_v2, synthetic first then real.

to run: open `Shireen_Baselines_v2.ipynb` in colab, set `ROOT` to wherever circuitree_v2 is unzipped, run all (~3 min)

- uses `graphs/<track>.npz` + `samples.csv`, no .bench parsing or pytorch graphs
- scored the same way as `evaluate.py` (checked, same numbers)
- WL rounds picked on val, test only run once at the end
- `baselines_v2.py` is the same code if you want to import it

## results on test (Hit@1 / mAP)

| method | synthetic hard | synthetic full | real hard | real full |
|---|---|---|---|---|
| histogram | 0.42 / 0.36 | 0.23 / 0.11 | 0.61 / 0.60 | 0.37 / 0.16 |
| WL (h=2) | 0.82 / 0.53 | 0.71 / 0.30 | 0.77 / 0.69 | 0.49 / 0.25 |
| WL + input positions (h=1) | 0.86 / 0.57 | 0.77 / 0.35 | 0.80 / 0.74 | 0.68 / 0.43 |
| dataset's wl_kernel (to compare) | 0.83 / 0.53 | 0.67 / 0.25 | 0.77 / 0.70 | 0.44 / 0.21 |

## time

| | per circuit | per search (1 query vs whole test set) |
|---|---|---|
| histogram | ~0.004 ms | ~2 ms |
| WL | ~0.06 ms | ~2.5 ms |
| CEC (ABC) | ~50-140 ms per pair (changes run to run) | ~8-20 min on real, ~1 min on synthetic |

CEC got 200/200 pairs right so it's exact, just way too slow to search with. all the numbers are in `shireen_baseline_results.csv`
