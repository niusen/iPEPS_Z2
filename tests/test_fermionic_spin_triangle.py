import os
import tempfile
import unittest
from collections import OrderedDict

import torch
import yastn

from ansatz.fermionic_spin_triangle_iPESS import (
    fermionic_spin_bond_dimension,
    fermionic_z2_config,
    gutzwiller_project_d4_sets,
    load_fermionic_spin_triangle_iPESS,
    make_fermionic_spin_state,
    save_fermionic_spin_triangle_iPESS,
)
from model.fermionic_spin_ob_iPESS import (
    spin_half_components_fermionic_d2,
    spin_half_operators_fermionic_d2,
)


class FermionicSpinTriangleTests(unittest.TestCase):
    def setUp(self):
        self.kwargs = {
            "backend": "torch",
            "default_dtype": "complex128",
            "default_device": "cpu",
            "Lx": 1,
            "Ly": 1,
        }

    def _old_d4_sets(self):
        config = fermionic_z2_config(self.kwargs)
        virtual = [
            yastn.Leg(config, s=signature, t=(0, 1), D=(1, 1))
            for signature in (1, 1, -1)
        ]
        physical = yastn.Leg(config, s=1, t=(0, 1), D=(2, 2))
        B = yastn.rand(config=config, legs=virtual)
        T = yastn.rand(
            config=config,
            legs=[virtual[2].conj(), physical, virtual[0].conj(), virtual[1].conj()],
        )
        return OrderedDict({"1,1": B}), OrderedDict({"1,1": T})

    def test_projection_copies_only_odd_physical_blocks(self):
        B_old, T_old = self._old_d4_sets()
        B_new, T_new = gutzwiller_project_d4_sets(B_old, T_old, self.kwargs)
        physical = T_new["1,1"].get_legs(axes=1)
        self.assertEqual(tuple(physical.t), ((1,),))
        self.assertEqual(tuple(physical.D), (2,))
        for charges in T_new["1,1"].get_blocks_charge():
            self.assertEqual(charges[1], 1)
            self.assertTrue(
                torch.equal(T_new["1,1"][charges], T_old["1,1"][charges])
            )
        self.assertTrue(torch.equal(B_new["1,1"].to_dense(), B_old["1,1"].to_dense()))

    def test_d2_json_roundtrip(self):
        B_old, T_old = self._old_d4_sets()
        B_set, T_set = gutzwiller_project_d4_sets(B_old, T_old, self.kwargs)
        state = make_fermionic_spin_state(B_set, T_set, self.kwargs)
        self.assertEqual(fermionic_spin_bond_dimension(state), 2)
        with tempfile.TemporaryDirectory() as directory:
            prefix = os.path.join(directory, "projected")
            save_fermionic_spin_triangle_iPESS(B_set, T_set, prefix, self.kwargs)
            B_load, T_load = load_fermionic_spin_triangle_iPESS(prefix, self.kwargs)
        for original, loaded in ((B_set["1,1"], B_load["1,1"]), (T_set["1,1"], T_load["1,1"])):
            self.assertEqual(original.get_blocks_charge(), loaded.get_blocks_charge())
            for charges in original.get_blocks_charge():
                self.assertTrue(torch.equal(original[charges], loaded[charges]))

    def test_spin_operators_are_physical_spin_half(self):
        Sx, Sy, Sz = spin_half_components_fermionic_d2(self.kwargs)
        expected = (
            0.5 * torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
            0.5 * torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
            0.5 * torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128),
        )
        for operator, matrix in zip((Sx, Sy, Sz), expected):
            self.assertTrue(torch.allclose(operator.to_dense(), matrix))

        Sa, Sb, _, S1, S2, S3, _, _ = spin_half_operators_fermionic_d2(
            self.kwargs
        )
        SS_factorized = yastn.ncon([Sa, Sb], [[1, -1, -2], [1, -3, -4]])
        SS_exact = (
            yastn.ncon([Sx, Sx], [[-1, -2], [-3, -4]])
            + yastn.ncon([Sy, Sy], [[-1, -2], [-3, -4]])
            + yastn.ncon([Sz, Sz], [[-1, -2], [-3, -4]])
        )
        self.assertLess(float(yastn.linalg.norm(SS_factorized - SS_exact)), 1.0e-12)

        chirality_factorized = yastn.ncon(
            [S1, S2, S3], [[-1, -4, 1], [1, -2, -5, 2], [2, -3, -6]]
        )
        chirality_exact = (
            yastn.ncon([Sx, Sy, Sz], [[-1, -4], [-2, -5], [-3, -6]])
            - yastn.ncon([Sx, Sz, Sy], [[-1, -4], [-2, -5], [-3, -6]])
            + yastn.ncon([Sy, Sz, Sx], [[-1, -4], [-2, -5], [-3, -6]])
            - yastn.ncon([Sy, Sx, Sz], [[-1, -4], [-2, -5], [-3, -6]])
            + yastn.ncon([Sz, Sx, Sy], [[-1, -4], [-2, -5], [-3, -6]])
            - yastn.ncon([Sz, Sy, Sx], [[-1, -4], [-2, -5], [-3, -6]])
        )
        relative_error = yastn.linalg.norm(chirality_factorized - chirality_exact)
        relative_error = relative_error / yastn.linalg.norm(chirality_exact)
        self.assertLess(float(relative_error), 1.0e-12)


if __name__ == "__main__":
    unittest.main()
