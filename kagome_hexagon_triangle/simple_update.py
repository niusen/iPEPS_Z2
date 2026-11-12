"""Dense blocked two-site SU, exact five-spin gates and QR/SVD truncation.

Hamiltonian convention here is the paper's H=sum S.S-h*sum Sz (h>0).
Each coarse edge gets four AB bonds, two A-internal bonds, and fractions
of the local fields. These nine edge Hamiltonians sum to the full cell
Hamiltonian exactly. This allocation and the Trotter schedule are our
explicit choices: the paper does not publish these implementation details.
"""
from dataclasses import dataclass
from functools import lru_cache
import torch
from .geometry import edge_support, neighbour

SZ = torch.diag(torch.tensor([.5, -.5], dtype=torch.float64))
SP = torch.tensor([[0., 1.], [0., 0.]], dtype=torch.float64)
SM = SP.T.contiguous()
ID = torch.eye(2, dtype=torch.float64)


def embed(op, i, n):
    out = torch.ones((1, 1), dtype=op.dtype, device=op.device)
    for j in range(n):
        out = torch.kron(out, op if i == j else ID.to(op))
    return out


@lru_cache(None)
def edge_hamiltonian(k, field):
    aa, bb, internal, bonds = edge_support(k)
    operators = [[embed(o, i, 5) for i in range(5)] for o in (SZ, SP, SM)]
    h = torch.zeros((32, 32), dtype=torch.float64)
    local_bonds = [(aa.index(i), aa.index(j)) for i, j in internal]
    local_bonds += [(aa.index(i), 3+bb.index(j)) for i, j in bonds]
    for i, j in local_bonds:
        h += operators[0][i]@operators[0][j]
        h += .5*(operators[1][i]@operators[2][j]+operators[2][i]@operators[1][j])
    for i in aa:
        # Even-numbered hexagon sites occur in two edge supports, odd in one.
        h -= field*(.5 if i % 2 == 0 else 1.)*operators[0][aa.index(i)]
    for j in bb:
        h -= .5*field*operators[0][3+bb.index(j)]
    return h


@lru_cache(None)
def gate_factors(k, field, dt):
    gate = torch.matrix_exp(-dt*edge_hamiltonian(k, field))
    # (Aout=8,Bout=4,Ain=8,Bin=4) -> operator Schmidt decomposition.
    u, s, vh = torch.linalg.svd(gate.reshape(8, 4, 8, 4).permute(0, 2, 1, 3).reshape(64, 16), full_matrices=False)
    keep = s > 1e-14*s[0]
    ga = (u[:, keep]*s[keep].sqrt()).T.reshape(-1, 8, 8)
    gb = (vh[keep]*s[keep].sqrt()[:, None]).reshape(-1, 4, 4)
    return ga, gb


def apply_local_factors(t, ops, sites, n):
    # Return [physical, other virtual indices..., Schmidt operator index].
    inactive = [i for i in range(n) if i not in sites]
    perm = sites+inactive+list(range(n, n+t.ndim-1))
    x = t.reshape(*([2]*n), *t.shape[1:]).permute(perm)
    rest = x.shape[len(sites):]
    x = x.reshape(2**len(sites), -1)
    x = torch.einsum('sij,jv->ivs', ops, x).reshape(*([2]*n), *t.shape[1:], len(ops))
    order = [sites.index(i) if i in sites else len(sites)+inactive.index(i) for i in range(n)]
    return x.permute(order+list(range(n, x.ndim))).reshape(*t.shape, len(ops))


@dataclass
class BlockState:
    A: list
    B: list
    weights: list  # weights[c][k] on A_c -- B_neighbour(c,k)
    field: float
    seed: int
    steps: int = 0
    c3: bool = False

    @property
    def D(self):
        return self.A[0].shape[1]

    @classmethod
    def random(cls, D, field=.38, seed=1701, c3=False):
        gen = torch.Generator().manual_seed(seed)
        arrays = [[torch.randn((p, D, D, D), generator=gen, dtype=torch.float64) for _ in range(3)] for p in (64, 8)]
        arrays = [[x/x.norm() for x in group] for group in arrays]
        w = [[torch.ones(D, dtype=torch.float64)/D**.5 for _ in range(3)] for _ in range(3)]
        st = cls(*arrays, w, field, seed, c3=c3)
        if c3:
            copy_rotation_orbit(st, 'A', 0)
            copy_rotation_orbit(st, 'B', 0)
        return st

    def lambdas(self, kind, c):
        if kind == 'A':
            return self.weights[c]
        # inverse of c_B = c_A + shift_n - shift_m
        return [self.weights[(c-(neighbour(0, k)))%3][k] for k in range(3)]

    def dressed(self, kind, c, power=.5):
        t = (self.A if kind == 'A' else self.B)[c]
        for k, w in enumerate(self.lambdas(kind, c)):
            shape = [1]*4
            shape[k+1] = len(w)
            t = t*w.pow(power).reshape(shape)
        return t/t.norm()

    def save(self, path, **metadata):
        torch.save(dict(format='kagome_hexagon_triangle_v1', A=self.A, B=self.B,
                        weights=self.weights, field=self.field, seed=self.seed,
                        steps=self.steps, c3=self.c3, metadata=metadata), path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location='cpu', weights_only=False)
        if d['format'] != 'kagome_hexagon_triangle_v1':
            raise ValueError('Different ansatz/checkpoint convention')
        return cls(**{k:d[k] for k in ('A', 'B', 'weights', 'field', 'seed', 'steps')}, c3=d.get('c3', False))


def rotate_tensor(t, kind):
    """Clockwise 120 degree spatial rotation, INCLUDING physical-spin axes.

    About a central coarse hexagonal face: A_c -> A_(c+1), B_c -> B_(c-1),
    virtual leg k -> k-1. A physical vertex i -> i-2; B vertex j -> j-1.
    No spin-space rotation or projection is involved.
    """
    n = 6 if kind == 'A' else 3
    shift = 2 if kind == 'A' else 1
    order = [(i+shift)%n for i in range(n)]+[n+1, n+2, n]
    return t.reshape(*([2]*n), *t.shape[1:]).permute(order).reshape_as(t).contiguous()


def copy_rotation_orbit(state, kind, c):
    group = state.A if kind == 'A' else state.B
    delta = 1 if kind == 'A' else -1
    group[(c+delta)%3] = rotate_tensor(group[c], kind)
    group[(c+2*delta)%3] = rotate_tensor(group[(c+delta)%3], kind)


@torch.no_grad()
def update_bond(state, c, k, dt):
    b = neighbour(c, k)
    order = [0]+[j+1 for j in range(3) if j != k]+[k+1]
    inv = [order.index(j) for j in range(4)]
    a = state.A[c].clone()
    t = state.B[b].clone()
    for kind, q, x in (('A', c, a), ('B', b, t)):
        for leg, w in enumerate(state.lambdas(kind, q)):
            if leg != k:
                shape = [1]*4
                shape[leg+1] = len(w)
                x *= w.reshape(shape)
    a, t = a.permute(order), t.permute(order)
    ga, gb = gate_factors(k, state.field, float(dt))
    aa, bb, _, _ = edge_support(k)
    x = apply_local_factors(a, ga, aa, 6)
    y = apply_local_factors(t, gb, bb, 3)
    w = state.weights[c][k].sqrt()
    # Exact evolved two-block matrix = X @ Y.T. QR compresses its columns
    # before SVD, avoiding a large (64*D^2) x (8*D^2) dense SVD.
    x = (x*w.reshape(1, 1, 1, -1, 1)).flatten(-2).reshape(64*state.D**2, -1)
    y = (y*w.reshape(1, 1, 1, -1, 1)).flatten(-2).reshape(8*state.D**2, -1)
    qx, rx = torch.linalg.qr(x, mode='reduced')
    qy, ry = torch.linalg.qr(y, mode='reduced')
    u, s, vh = torch.linalg.svd(rx@ry.T, full_matrices=False)
    discarded = float(s[state.D:].square().sum()/s.square().sum())
    a = (qx@u[:, :state.D]).reshape(64, state.D, state.D, state.D).permute(inv)
    t = (qy@vh[:state.D].T).reshape(8, state.D, state.D, state.D).permute(inv)
    new_w = s[:state.D]/s[:state.D].norm()
    change = float((new_w-state.weights[c][k]).abs().max())
    for kind, q, x in (('A', c, a), ('B', b, t)):
        for leg, w in enumerate(state.lambdas(kind, q)):
            if leg != k:
                shape = [1]*4
                shape[leg+1] = len(w)
                x /= w.clamp_min(1e-12).reshape(shape)
    state.A[c], state.B[b] = a/a.norm(), t/t.norm()
    state.weights[c][k] = new_w
    return change, discarded


@torch.no_grad()
def sweep(state, dt, second_order=True):
    # In C3 mode the orbit of (0,k) comprises three disjoint two-block
    # gates. Updating one and copying its rotations is the exact same SU
    # operation on that matching, without three redundant factorizations.
    bonds = [(0, k) for k in range(3)] if state.c3 else [(c, k) for k in range(3) for c in range(3)]
    changes, errors = [], []
    # Symmetric palindrome ensures a second-order Trotter step. The paper
    # only specifies two-site gates; its exact splitting is not published.
    sequence = bonds+bonds[::-1] if second_order else bonds
    for c, k in sequence:
        change, error = update_bond(state, c, k, dt/2 if second_order else dt)
        if state.c3:
            copy_rotation_orbit(state, 'A', c)
            copy_rotation_orbit(state, 'B', neighbour(c, k))
            state.weights[(c+1)%3][(k-1)%3] = state.weights[c][k].clone()
            state.weights[(c+2)%3][(k-2)%3] = state.weights[c][k].clone()
        changes.append(change)
        errors.append(error)
    state.steps += 1
    return max(changes), max(errors)
