# FGNN2 fixes and speedups

## 1. pretrain.py does not run as-is (bug)
Line 1 is `from dataset_gcl import *`, but the author deleted `dataset_gcl.py`
(commit 0fca36b). `tee` was coming in through that star import.
Fix: replace line 1 with

    import tee

## 2. README command problems
- `--datapath ../dataset/global_data/` should be `../datasets/global_data/`
- the backticks around the start point make the shell try to *run* it. Also the code
  already adds `../checkpoints/` in front, so pass the path relative to that:

      python train_global.py --datapath ../datasets/global_data/ --batch_size 128 --nlabels 4 \
        --ratio 0.5 --flag_inv --pre_train --start_point example/weight.pth \
        --checkpoint global/example

## 3. Speed: batch the graphs in train_global.py
`train()` and `test()` run the GNN on one circuit at a time and grow `embeddings` with
`th.cat` inside the loop. Since `forward_global` loops over topological levels, many
small graphs cost (number of graphs x depth) steps. With `dgl.batch` it becomes just
(max depth) steps for the whole batch. Use `embed_batch()` from eval_retrieval.py
(pass the stored PO_nids, and po_inv = zeros because this dataset already stores `inv`
on the nodes).

## 4. Speed: vectorized NCE loss in pretrain.py
See fast_nce.py (one matrix op instead of a Python loop over pairs).

## 5. Fair numbers: choose the epoch with validation, not test
train_global.py prints test accuracy every epoch. Picking the best test epoch inflates
accuracy. Keep the epoch with the best *val* accuracy and report test for that one only.
