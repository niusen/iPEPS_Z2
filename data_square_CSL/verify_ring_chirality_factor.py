import itertools
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import torch

from model.bosonic_square_ob_iPEPS import (
    RING_EXCHANGE_CHIRALITY_FACTOR,
    prl_129_177201_square_csl_parameters,
)


dtype = torch.complex128
device = "cpu"

Id = torch.eye(2, dtype=dtype, device=device)
sx = 0.5 * torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype, device=device)
sy = 0.5 * torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=dtype, device=device)
sz = 0.5 * torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype, device=device)
spins = (sx, sy, sz)


def kron_ops(ops):
    out = ops[0]
    for op in ops[1:]:
        out = torch.kron(out, op)
    return out


def op_on(site, op):
    return kron_ops([op if ind == site else Id for ind in range(4)])


S = [[op_on(site, op) for op in spins] for site in range(4)]


def swap(i, j):
    out = 0.5 * torch.eye(16, dtype=dtype, device=device)
    for a in range(3):
        out = out + 2.0 * S[i][a] @ S[j][a]
    return out


def levi_civita(a, b, c):
    if len({a, b, c}) < 3:
        return 0
    inversions = sum(1 for i, j in itertools.combinations((a, b, c), 2) if i > j)
    return -1 if inversions % 2 else 1


def chirality(i, j, k):
    out = torch.zeros((16, 16), dtype=dtype, device=device)
    for a, b, c in itertools.product(range(3), repeat=3):
        eps = levi_civita(a, b, c)
        if eps:
            out = out + eps * S[i][a] @ S[j][b] @ S[k][c]
    return out


P12 = swap(0, 1)
P23 = swap(1, 2)
P34 = swap(2, 3)

ring_exchange = 1.0j * (P12 @ P23 @ P34 - P34 @ P23 @ P12)
chirality_sum = (
    chirality(0, 1, 2)
    + chirality(1, 2, 3)
    + chirality(2, 3, 0)
    + chirality(3, 0, 1)
)

residual = torch.linalg.norm(ring_exchange - RING_EXCHANGE_CHIRALITY_FACTOR * chirality_sum)
print("ring_exchange_chirality_factor:", RING_EXCHANGE_CHIRALITY_FACTOR)
print("identity residual:", residual.item())
print("paper parameters:", prl_129_177201_square_csl_parameters(chirality_sign=1.0))

if residual.item() > 1.0e-12:
    raise RuntimeError("ring-exchange/chirality identity check failed")
