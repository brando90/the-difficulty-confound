"""Many independent copies of Experiment 11's 104,002-parameter transformer, trained side by side in one process.

Every parameter of Experiment 11's model.Transformer is stored once per model, stacked on a leading model axis, under
the same state_dict name. A batch holds one string length per model group (as in Experiment 11, where every batch has
a single length), so there is no padding and the prediction is read at the last position exactly as model.py does.

Public API: init_stack, forward, export_one, load_one, clip_per_model_, n_models.
"""
import math

import torch
import torch.nn.functional as F

LN_EPS = 1e-5  # nn.LayerNorm default, as in model.py


def init_stack(ref_models, device):
    """Stack the state_dicts of Experiment 11 models (one per ensemble member) into leaf tensors on `device`."""
    sds = [m.state_dict() for m in ref_models]
    return {k: torch.stack([sd[k].detach().float() for sd in sds]).to(device).requires_grad_(True) for k in sds[0]}


def n_models(P):
    return P['tok.weight'].shape[0]


def _ln(x, w, b):
    return F.layer_norm(x, x.shape[-1:], eps=LN_EPS) * w[:, None, :] + b[:, None, :]


def _lin(x, W, b):
    """x (R, N, i), W (R, o, i), b (R, o) -> (R, N, o)."""
    return torch.baddbmm(b[:, None, :], x, W.transpose(1, 2))


def forward(P, x, n_layers, n_heads):
    """x: LongTensor (R, B, L) of tokens, one sequence length for all. Returns 2-way logits (R, B)."""
    R, B, L = x.shape
    d = P['tok.weight'].shape[-1]
    ridx = torch.arange(R, device=x.device)[:, None, None]
    h = P['tok.weight'][ridx, x] + P['pos.weight'][:, None, :L, :]
    h = h.reshape(R, B * L, d)
    dh = d // n_heads
    for i in range(n_layers):
        p = f'blocks.{i}.'
        qkv = _lin(_ln(h, P[p + 'ln1.weight'], P[p + 'ln1.bias']), P[p + 'qkv.weight'], P[p + 'qkv.bias'])
        q, k, v = (t.reshape(R * B, L, n_heads, dh).transpose(1, 2) for t in qkv.split(d, dim=-1))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(R, B * L, d)
        h = h + _lin(a, P[p + 'proj.weight'], P[p + 'proj.bias'])
        m = F.gelu(_lin(_ln(h, P[p + 'ln2.weight'], P[p + 'ln2.bias']), P[p + 'fc1.weight'], P[p + 'fc1.bias']))
        h = h + _lin(m, P[p + 'fc2.weight'], P[p + 'fc2.bias'])
    last = h.reshape(R, B, L, d)[:, :, -1, :]
    return _lin(_ln(last, P['ln_f.weight'], P['ln_f.bias']), P['head.weight'], P['head.bias'])


@torch.no_grad()
def clip_per_model_(P, max_norm):
    """torch.nn.utils.clip_grad_norm_ applied to each model separately (its own global L2 norm over all params)."""
    sq = None
    for t in P.values():
        s = t.grad.float().pow(2).reshape(t.shape[0], -1).sum(1)
        sq = s if sq is None else sq + s
    norm = sq.sqrt()
    coef = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
    for t in P.values():
        t.grad.mul_(coef.view(-1, *([1] * (t.dim() - 1))).to(t.grad.dtype))
    return norm


def export_one(P, r):
    """float32 numpy arrays of model r keyed by Experiment 11 state_dict names (loadable with model.load_npz)."""
    return {k: v[r].detach().float().cpu().numpy() for k, v in P.items()}


def load_one(model, arrays):
    model.load_state_dict({k: torch.from_numpy(arrays[k].copy()) for k in arrays})
    return model


def bits_from_logits(logits, y):
    """-log2 p(correct), (R, B) float32 logits and int64 labels."""
    return -F.log_softmax(logits.float(), dim=-1).gather(-1, y[..., None])[..., 0] / math.log(2.0)
