"""Adapters from the dense triangular iPESS/CTM objects to generic iPEPS EH.

Only this module is specific to iPEPS_Z2.  The entanglement-spectrum matvec in
``generic_eh_momentum_stream.py`` keeps the original tn-torch_dev algorithm.
"""

from collections import OrderedDict

import torch
import yastn


def _cell_size(global_args):
    if isinstance(global_args, dict):
        return global_args["Lx"], global_args["Ly"]
    return global_args.Lx, global_args.Ly


def ipess_to_ipeps_yastn(B_set, T_set, global_args):
    """Contract ``B=(L,U,M)`` and ``T=(M,s,R,D)`` into ``A=(L,D,R,U,s)``."""
    Lx, Ly = _cell_size(global_args)
    A_set = OrderedDict()
    for cx in range(1, Lx + 1):
        for cy in range(1, Ly + 1):
            key = str(cx) + "," + str(cy)
            B, T = B_set[key], T_set[key]
            if B.config.fermionic or T.config.fermionic:
                raise ValueError(
                    "This generic EH adapter is for the dense bosonic spin iPESS. "
                    "A fermionic Z2 state requires a graded boundary translation."
                )
            # intermediate order: (L,U,s,R,D)
            A = yastn.ncon([B, T], [[-1, -2, 1], [1, -3, -4, -5]])
            # Native square-iPEPS order used in iPEPS_Z2: (L,D,R,U,s)
            A_set[key] = yastn.transpose(A, axes=(0, 4, 3, 1, 2))
    return A_set


class DenseIPEPSState:
    """Minimal original tn-torch_dev ``IPEPS`` interface for the EH code.

    Stored torch tensors use the original order ``(physical,up,left,down,right)``.
    Coordinates are zero based, as in tn-torch_dev.
    """

    def __init__(self, A_set, global_args):
        self.lX, self.lY = _cell_size(global_args)
        self.sites = OrderedDict()
        for cx in range(1, self.lX + 1):
            for cy in range(1, self.lY + 1):
                key = str(cx) + "," + str(cy)
                A = A_set[key].to_dense()
                # iPEPS_Z2 A=(L,D,R,U,s) -> tn-torch A=(s,U,L,D,R)
                self.sites[(cx - 1, cy - 1)] = A.permute(4, 3, 0, 1, 2).contiguous()
        first = next(iter(self.sites.values()))
        self.dtype = first.dtype
        self.device = first.device

    def vertexToSite(self, coord):
        return (coord[0] % self.lX, coord[1] % self.lY)

    def site(self, coord):
        return self.sites[self.vertexToSite(coord)]

    def get_aux_bond_dims(self):
        return [dim for site in self.sites.values() for dim in site.shape[1:]]


class DenseCTMEnv:
    """Expose iPEPS_Z2 CTM edges using the original generic ``ENV.T`` API."""

    def __init__(self, CTM_cell, state):
        self.T = {}
        self.C = {}
        self.dtype = state.dtype
        self.device = state.device
        inferred_chi = None
        for x in range(state.lX):
            for y in range(state.lY):
                coord = (x, y)
                key = str(x + 1) + "," + str(y + 1)
                edges = CTM_cell["Tset"][key]
                T1 = edges["T1"].to_dense().contiguous()
                T2 = edges["T2"].to_dense().contiguous()
                T3 = edges["T3"].to_dense().contiguous()
                T4 = edges["T4"].to_dense().contiguous()

                # Original generic ENV raw layouts:
                # top/right=(chi,D^2,chi), left=(chi,chi,D^2),
                # bottom=(D^2,chi,chi).
                self.T[(coord, (0, -1))] = T1
                self.T[(coord, (1, 0))] = T2
                self.T[(coord, (0, 1))] = T3.permute(1, 0, 2).contiguous()
                self.T[(coord, (-1, 0))] = T4.permute(0, 2, 1).contiguous()

                local_chi = T2.shape[0]
                inferred_chi = local_chi if inferred_chi is None else inferred_chi
                if local_chi != inferred_chi:
                    raise ValueError("CTM boundary chi is not uniform across the cell")

        self.chi = inferred_chi
        self._validate(state)

    def _validate(self, state):
        for coord in state.sites:
            site = state.site(coord)
            expected = {
                (0, -1): (self.chi, site.size(1) ** 2, self.chi),
                (-1, 0): (self.chi, self.chi, site.size(2) ** 2),
                (0, 1): (site.size(3) ** 2, self.chi, self.chi),
                (1, 0): (self.chi, site.size(4) ** 2, self.chi),
            }
            for direction, shape in expected.items():
                actual = tuple(self.T[(coord, direction)].shape)
                if actual != shape:
                    raise ValueError(
                        "CTM edge layout mismatch at " + str((coord, direction))
                        + ": expected " + str(shape) + ", found " + str(actual)
                    )


def build_dense_ipeps_and_env(B_set, T_set, CTM_cell, global_args):
    A_set = ipess_to_ipeps_yastn(B_set, T_set, global_args)
    state = DenseIPEPSState(A_set, global_args)
    env = DenseCTMEnv(CTM_cell, state)
    return A_set, state, env


def conversion_double_layer_relative_error(B_set, T_set, A_set):
    """Check that combined-iPESS and converted-iPEPS double layers coincide."""
    from ansatz.square_iPEPS import build_double_layer
    from ctmrg.CTMRG_unitcell_iPESS import build_doublelayer_iPESS

    errors = {}
    for key, A in A_set.items():
        cx, cy = (int(value) for value in key.split(","))
        T_double, B_double = build_doublelayer_iPESS(B_set, T_set, (cx, cy))
        AA_ipess = yastn.ncon(
            [B_double, T_double], [[-1, 1, -4], [-2, -3, 1]]
        )
        AA_ipeps = build_double_layer(A.conj(), A)
        denominator = max(float(yastn.linalg.norm(AA_ipess)), 1.0e-300)
        errors[key] = float(yastn.linalg.norm(AA_ipess - AA_ipeps)) / denominator
    return errors
