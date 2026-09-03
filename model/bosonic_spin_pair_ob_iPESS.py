"""Two decoupled triangular J1-Jchi layers with opposite Jchi signs.

The onsite physical leg has dimension four.  A single fixed unitary maps it
to two spin-1/2 legs.  The unitary and every operator derived from it are
built once in :class:`TriangleSpinChiralPairModel`; energy and line-search
calls reuse that same model object.
"""

import torch
from torch.utils.checkpoint import checkpoint

from model.Spin_ob_iPESS import (
    hopping_diagonala_iPESS_no_sign,
    hopping_x_iPESS_no_sign,
    hopping_y_iPESS_no_sign,
    ob_dn_triangle_iPESS,
    ob_onsite_iPESS,
    ob_up_triangle_iPESS,
)
from model.bosonic_spin_ob_iPESS import (
    _matrix_to_yastn,
    factorized_spin_operators_dense,
)


def fixed_d4_to_two_spin_unitary(dtype=torch.complex128, device="cpu"):
    r"""Return the fixed map from ``p`` to ``(a,b)`` with ``p=2*a+b``.

    Rows are the product basis ``(a,b)`` and columns are the d=4 physical
    basis.  Thus an operator in the product basis is represented on the
    physical leg as ``U^dagger O U``.  The identity is intentional: keeping
    it as an explicit matrix makes the split convention testable and fixed.
    """
    return torch.eye(4, dtype=dtype, device=device)


def _spin_half_matrices(dtype, device):
    sx = 0.5 * torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    sy = 0.5 * torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    sz = 0.5 * torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    return sx, sy, sz


class TriangleSpinChiralPairModel:
    r"""Fixed-basis d=4 model for H_A(Jchi) + H_B(-Jchi)."""

    model_name = "triangle_spin_chiral_pair_J1_Jchi"

    def __init__(self, config_kwargs, split_unitary):
        device = config_kwargs.get("default_device", "cpu")
        dtype_name = config_kwargs.get("default_dtype", "complex128")
        dtype = torch.complex64 if dtype_name == "complex64" else torch.complex128
        unitary = torch.as_tensor(split_unitary, dtype=dtype, device=device).detach().clone()
        if tuple(unitary.shape) != (4, 4):
            raise ValueError("split_unitary must be a 4x4 matrix")
        identity = torch.eye(4, dtype=dtype, device=device)
        error = torch.linalg.norm(unitary.conj().T @ unitary - identity)
        if float(error.detach().cpu()) > 1.0e-12:
            raise ValueError("split_unitary is not unitary")

        # This object owns the only U used during the full optimization.
        self.split_unitary = unitary
        self.config_kwargs = dict(config_kwargs)
        self.dtype = dtype
        self.device = torch.device(device)

        sx, sy, sz = _spin_half_matrices(dtype, device)
        eye2 = torch.eye(2, dtype=dtype, device=device)
        self.components = {}
        self.factorized = {}
        for layer in ("chiral", "antichiral"):
            product_ops = (
                (torch.kron(sx, eye2), torch.kron(sy, eye2), torch.kron(sz, eye2))
                if layer == "chiral"
                else (torch.kron(eye2, sx), torch.kron(eye2, sy), torch.kron(eye2, sz))
            )
            physical_ops = tuple(
                unitary.conj().T @ operator @ unitary for operator in product_ops
            )
            yastn_ops = tuple(
                _matrix_to_yastn(operator, config_kwargs) for operator in physical_ops
            )
            self.components[layer] = yastn_ops
            self.factorized[layer] = factorized_spin_operators_dense(*yastn_ops)

    def unitary_fingerprint(self):
        """A deterministic log value that identifies the fixed split basis."""
        weights = torch.arange(1, 17, dtype=torch.float64, device=self.device).reshape(4, 4)
        value = torch.sum(self.split_unitary.real.to(torch.float64) * weights)
        value = value + 1j * torch.sum(self.split_unitary.imag.to(torch.float64) * weights)
        return complex(value.detach().cpu().item())

    @staticmethod
    def _couplings(parameters):
        J1 = parameters.get("J1", 1.0)
        if "Jchi_chiral" not in parameters or "Jchi_antichiral" not in parameters:
            raise KeyError(
                "chiral-pair parameters must explicitly contain both "
                "Jchi_chiral and Jchi_antichiral"
            )
        jchi_chiral = parameters["Jchi_chiral"]
        jchi_antichiral = parameters["Jchi_antichiral"]
        scale = max(1.0, abs(float(jchi_chiral)), abs(float(jchi_antichiral)))
        if abs(float(jchi_chiral) + float(jchi_antichiral)) > 1.0e-14 * scale:
            raise ValueError(
                "the chiral pair requires Jchi_antichiral == -Jchi_chiral"
            )
        return J1, {"chiral": jchi_chiral, "antichiral": jchi_antichiral}

    def evaluate(
        self,
        parameters,
        B_set,
        T_set,
        double_B_set,
        double_T_set,
        CTM_cell,
        global_args,
        return_observables=False,
    ):
        """Evaluate the average energy per layer and per spatial site."""
        J1, jchi = self._couplings(parameters)
        Lx, Ly = global_args.Lx, global_args.Ly
        names = ("chi_up", "chi_down", "SS_x", "SS_y", "SS_diagonal")
        values = {
            f"{name}_{layer}": []
            for layer in ("chiral", "antichiral")
            for name in names
        }
        if return_observables:
            values.update({
                f"S{axis}_{layer}": []
                for layer in ("chiral", "antichiral")
                for axis in ("x", "y", "z")
            })
        layer_energy = {
            "chiral": torch.zeros((), dtype=self.dtype, device=self.device),
            "antichiral": torch.zeros((), dtype=self.dtype, device=self.device),
        }

        for cx in range(1, Lx + 1):
            for cy in range(1, Ly + 1):
                for layer in ("chiral", "antichiral"):
                    Sa, Sb, SS_string, S1, S2, S3, string12, string23 = self.factorized[layer]
                    chi_up = checkpoint(
                        ob_up_triangle_iPESS,
                        CTM_cell, S1, S2, S3, string12, string23,
                        B_set, T_set, double_B_set, double_T_set,
                        cx, cy, Lx, Ly, use_reentrant=False,
                    )
                    # Both layers use the same common spatial orientation.
                    chi_down = -checkpoint(
                        ob_dn_triangle_iPESS,
                        CTM_cell, S1, S2, S3, string12, string23,
                        B_set, T_set, double_B_set, double_T_set,
                        cx, cy, Lx, Ly, use_reentrant=False,
                    )
                    ss_x = checkpoint(
                        hopping_x_iPESS_no_sign,
                        CTM_cell, Sa, Sb, SS_string, B_set, T_set,
                        double_B_set, double_T_set, cx, cy, Lx, Ly,
                        use_reentrant=False,
                    )
                    ss_y = checkpoint(
                        hopping_y_iPESS_no_sign,
                        CTM_cell, Sa, Sb, SS_string, B_set, T_set,
                        double_B_set, double_T_set, cx, cy, Lx, Ly,
                        use_reentrant=False,
                    )
                    ss_diagonal = checkpoint(
                        hopping_diagonala_iPESS_no_sign,
                        CTM_cell, Sa, Sb, SS_string, B_set, T_set,
                        double_B_set, double_T_set, cx, cy, Lx, Ly,
                        use_reentrant=False,
                    )
                    layer_energy[layer] = layer_energy[layer] + J1 * (
                        ss_x + ss_y + ss_diagonal
                    ) + jchi[layer] * (chi_up + chi_down)
                    for name, value in (
                        ("chi_up", chi_up), ("chi_down", chi_down),
                        ("SS_x", ss_x), ("SS_y", ss_y),
                        ("SS_diagonal", ss_diagonal),
                    ):
                        values[f"{name}_{layer}"].append(value)

                    if return_observables:
                        with torch.no_grad():
                            for axis, operator in zip(("x", "y", "z"), self.components[layer]):
                                value = ob_onsite_iPESS(
                                    CTM_cell, operator, B_set, T_set,
                                    double_B_set, double_T_set,
                                    cx, cy, Lx, Ly,
                                )
                                values[f"S{axis}_{layer}"].append(value)

        normalization = Lx * Ly
        layer_energy = {
            layer: torch.real(value / normalization)
            for layer, value in layer_energy.items()
        }
        average_energy = 0.5 * (
            layer_energy["chiral"] + layer_energy["antichiral"]
        )
        if not return_observables:
            return average_energy
        observables = {
            name: torch.stack(tuple(items)).reshape(Lx, Ly)
            for name, items in values.items()
        }
        observables["energy_chiral"] = layer_energy["chiral"]
        observables["energy_antichiral"] = layer_energy["antichiral"]
        return average_energy, observables
