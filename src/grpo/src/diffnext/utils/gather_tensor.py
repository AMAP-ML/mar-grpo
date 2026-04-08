
import torch
from deepspeed import zero as ds_zero


def gather_tensor(t, device=None, dtype=None):
    need = (getattr(t, "numel", lambda: 1)() == 0) or hasattr(t, "ds_id")
    if ds_zero is not None and need:
        with ds_zero.GatheredParameters([t], modifier_rank=None):
            out = t.detach().clone()
    else:
        out = t.detach().clone()
    if device is not None: out = out.to(device)
    if dtype  is not None: out = out.to(dtype=dtype)
    return out