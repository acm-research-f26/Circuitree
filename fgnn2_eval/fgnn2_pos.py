"""
fgnn2_pos.py -- builds FGNN2, with optional add-ons:
  positions (Experiment B): input i starts with  ones + pos_emb[i]  instead of plain ones.
                            pos_emb starts at zero, so before training it equals pretrained FGNN2.
  input-count rule (Experiment C): see add_input_count below.
"""
import torch as th
from FunctionConv import FuncConv


def make_model(hidden_dim=256, positions=False, max_inputs=256, weights=None, device='cpu'):
    model = FuncConv(hidden_dim, hidden_dim, flag_proj=False, flag_inv=True)
    if positions:
        model.pos_emb = th.nn.Embedding(max_inputs + 1, hidden_dim, padding_idx=0)
        th.nn.init.zeros_(model.pos_emb.weight)
    model = model.to(device)
    if weights:
        state = th.load(weights, map_location=device, weights_only=True)
        missing = [k for k in model.state_dict() if k not in state]
        if missing:
            print('loaded weights; new parts start fresh:', sorted({k.split('.')[0] for k in missing}))
        model.load_state_dict(state, strict=not missing)
    return model


def add_input_count(emb, num_inputs, weight=10.0, max_inputs=256):
    """Input-count rule: append a scaled one-hot of the number of inputs, so circuits with a
    different number of inputs can never look close (they can't be twins in this dataset)."""
    import numpy as np
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    oh = np.zeros((len(emb), max_inputs + 1), dtype=emb.dtype)
    oh[np.arange(len(emb)), np.minimum(np.asarray(num_inputs), max_inputs)] = weight
    return np.hstack([emb, oh])
