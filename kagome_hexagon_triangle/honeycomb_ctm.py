"""Native 120-degree honeycomb CTMRG, three corners and six edges.

Index equations follow Nyckees et al., arXiv:2306.09046v2, III.2,
Figs.6--8, Eqs.8--13. Matrix direction is clockwise, hence transposed
relative to their drawing: C_i_new=U_(i-1)^H C_i_expanded U_i.

For the present 3-colour enlarged cell, the three outward face shifts are
u,-v,v-u. They all change c=n-m by +1 modulo three. Thus each new face
environment c uses the old environment and local tensors of face c+1.
All three environments must be updated simultaneously; using a single
environment for six independent tensors would contract the wrong state.

We do not impose C3 symmetry on arbitrary random tensors. The three-corner
geometry is retained; separate directions and face colours are kept.
"""
from dataclasses import dataclass
import opt_einsum as oe
import torch
from .geometry import face_nodes

ROTATIONS = ((0, 1, 2), (2, 0, 1), (1, 2, 0))


def contract(eq, *args):
    return oe.contract(eq, *args, backend='torch', optimize='auto-hq')


def normalized(t):
    return t/t.norm().clamp_min(1e-300)


def double_layer(t, op=None):
    # Fused legs (ket,bra), in the same global k=0,1,2 order as SU.
    bra = t.conj() if op is None else torch.einsum('pq,pabc->qabc', op.to(t), t.conj())
    d = t.shape[1]**2
    return torch.einsum('pabc,pdef->adbecf', t, bra).reshape(d, d, d)


def layers(state):
    return {(kind, c): double_layer(state.dressed(kind, c))
            for kind in ('A', 'B') for c in range(3)}


def face_pairs(local, c):
    nodes = face_nodes(c)
    aa = [local[nodes[2*i]].permute(ROTATIONS[i]) for i in range(3)]
    bb = [local[nodes[2*i+1]].permute(ROTATIONS[i]) for i in range(3)]
    # An edge Ta_i absorbs the A tensor clockwise after B_i, with the
    # orientation of direction i (not the next corner's orientation).
    ae = [local[nodes[(2*i+2)%6]].permute(ROTATIONS[i]) for i in range(3)]
    return aa, bb, ae


def expanded_corner(C, Ta, Tb_prev, a, b):
    x = contract('lxu,uv,vtr,xyk,ztk->lyrz', Tb_prev, C, Ta, a, b)
    return x.reshape(x.shape[0]*x.shape[1], x.shape[2]*x.shape[3])


@dataclass
class FaceEnvironment:
    C: list
    Ta: list
    Tb: list

    @classmethod
    def initial(cls, D, dtype=torch.float64):
        boundary = torch.eye(D, dtype=dtype).flatten()/D**.5
        return cls([torch.ones((1, 1), dtype=dtype) for _ in range(3)],
                   [boundary.reshape(1, D*D, 1).clone() for _ in range(3)],
                   [boundary.reshape(1, D*D, 1).clone() for _ in range(3)])

    def corners(self, local, c):
        aa, bb, _ = face_pairs(local, c)
        return [expanded_corner(self.C[i], self.Ta[i], self.Tb[(i-1)%3], aa[i], bb[i]) for i in range(3)]

    def grow(self, local, c, chi):
        aa, bb, ae = face_pairs(local, c)
        cp = [normalized(expanded_corner(self.C[i], self.Ta[i], self.Tb[(i-1)%3], aa[i], bb[i])) for i in range(3)]
        us = []
        for i in range(3):
            rho = cp[(i+1)%3]@cp[(i+2)%3]@cp[i]
            u, _, _ = torch.linalg.svd(rho, full_matrices=False)
            us.append(u[:, :min(chi, u.shape[1])])
        cc, ta, tb = [], [], []
        for i in range(3):
            cc.append(normalized(us[(i-1)%3].mH@cp[i]@us[i]))
            # Edge absorption and the same isometry as on its adjacent corner.
            ua = us[i].reshape(self.Tb[i].shape[0], ae[i].shape[0], -1)
            ub = us[i].reshape(self.Ta[i].shape[2], bb[i].shape[0], -1)
            ta.append(normalized(contract('lzL,zyx,lxr->Lyr', ua.conj(), ae[i], self.Tb[i])))
            tb.append(normalized(contract('lxr,zxk,rzK->lkK', self.Ta[i], bb[i], ub)))
        return FaceEnvironment(cc, ta, tb)

    def spectrum(self):
        x = self.C[0]@self.C[1]@self.C[2]
        s = torch.linalg.svdvals(x)
        return s/s.norm().clamp_min(1e-300)


@dataclass
class HoneycombEnvironment:
    faces: list
    iterations: int = 0
    converged: bool = False
    residual: float = float('inf')

    @classmethod
    def initial(cls, D):
        return cls([FaceEnvironment.initial(D) for _ in range(3)])

    @torch.no_grad()
    def run(self, local, chi=16, max_iterations=50, tolerance=1e-6, callback=None, min_iterations=20):
        history = []
        self.converged = False
        for it in range(max_iterations):
            self.faces = [self.faces[(c+1)%3].grow(local, (c+1)%3, chi) for c in range(3)]
            spectra = [e.spectrum() for e in self.faces]
            self.iterations += 1
            # The honeycomb growth changes face colour and parity. Compare
            # after six growths, avoiding false convergence or a false failure
            # caused by a legitimate three-/two-cycle of the finite patches.
            if len(history) >= 6 and all(x.shape == y.shape for x, y in zip(spectra, history[-6])):
                self.residual = max(float((x-y).abs().max()) for x, y in zip(spectra, history[-6]))
                if self.residual < tolerance and it+1 >= min_iterations:
                    self.converged = True
            history.append(spectra)
            if callback:
                callback(dict(iteration=self.iterations, residual=self.residual,
                              corner_dimension=self.faces[0].C[0].shape[0]))
            if self.converged:
                break
        return self


class FaceMeasurement:
    def __init__(self, env, local, c):
        self.env, self.local, self.c = env, local, c
        self.cp = env.corners(local, c)
        self.norm = torch.trace(self.cp[0]@self.cp[1]@self.cp[2])
        self.complements = [self.cp[(i+1)%3]@self.cp[(i+2)%3] for i in range(3)]
        self.impurity_corners = {}
        self.impurity_references = {}
        self.responses = None
        if not torch.isfinite(self.norm) or float(self.norm.real) <= 0:
            raise ArithmeticError(f'Invalid CTM norm {self.norm}; environment not trustworthy')

    def expectation(self, replacements):
        if len(replacements) == 1:
            # Differentiate only this small, fixed-environment contraction.
            # This is a linear functional of each local double layer, NOT
            # differentiation through CTMRG or a variational optimization.
            # Reusing the six response tensors avoids a new corner contraction
            # for every Sz or intra-block bond observable.
            if self.responses is None:
                with torch.enable_grad():
                    variables = {key:t.detach().requires_grad_() for key, t in self.local.items()}
                    cc = self.env.corners(variables, self.c)
                    norm = torch.trace(cc[0]@cc[1]@cc[2])
                    grad = torch.autograd.grad(norm, list(variables.values()))
                    self.responses = {key:g.detach()/norm.detach() for key, g in zip(variables, grad)}
            key, value = next(iter(replacements.items()))
            return float((self.responses[key]*value).sum().real)
        # Reuse unchanged corners. One- and two-block impurities require
        # modifying at most two expanded corners, never a rank-six patch.
        local = self.local | replacements
        nodes = face_nodes(self.c)
        aa, bb, _ = face_pairs(local, self.c)
        cp, changed = [], []
        for i in range(3):
            if nodes[2*i] in replacements or nodes[2*i+1] in replacements:
                key = (i, id(local[nodes[2*i]]), id(local[nodes[2*i+1]]))
                same_corner_pair = nodes[2*i] in replacements and nodes[2*i+1] in replacements
                if same_corner_pair:
                    cp.append(expanded_corner(self.env.C[i], self.env.Ta[i], self.env.Tb[(i-1)%3], aa[i], bb[i]))
                else:
                    if key not in self.impurity_corners:
                        self.impurity_corners[key] = expanded_corner(self.env.C[i], self.env.Ta[i], self.env.Tb[(i-1)%3], aa[i], bb[i])
                        # Keep references to prevent Python reusing tensor ids.
                        self.impurity_references[key] = (local[nodes[2*i]], local[nodes[2*i+1]])
                    cp.append(self.impurity_corners[key])
                changed.append(i)
            else:
                cp.append(self.cp[i])
        if len(changed) == 1:
            i = changed[0]
            value = torch.einsum('ij,ji->', cp[i], self.complements[i])/self.norm
        else:
            value = torch.trace(cp[0]@cp[1]@cp[2])/self.norm
        if value.is_complex() and abs(float(value.imag)) > 1e-8:
            raise ArithmeticError('Complex expectation of Hermitian observable')
        return float(value.real)
