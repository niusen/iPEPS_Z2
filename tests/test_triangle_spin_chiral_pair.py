import os
import tempfile
import unittest
from collections import OrderedDict

import numpy as np
import torch
import yastn

from ansatz.bosonic_triangle_iPESS import (
    bosonic_triangle_iPESS_bond_dimension,
    chiral_pair_from_single_bosonic_triangle_iPESS,
    random_bosonic_triangle_iPESS,
)
from config.config import (
    CTMARGS,
    GLOBALARGS,
    INITCTMARGS,
    LINESEARCH,
    Square_Hubbard_Energy_settings,
)
from ctmrg.CTMRG_unitcell_iPESS import CTMRG_cell_iPESS
from model.bosonic_spin_ob_iPESS import evaluate_triangle_spin_energy
from model.bosonic_spin_pair_ob_iPESS import (
    TriangleSpinChiralPairModel,
    fixed_d4_to_two_spin_unitary,
)
from optimization.stochastic_opt import get_grad, optimize_iPESS


class TriangleSpinChiralPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = {
            "backend": "torch",
            "default_dtype": "complex128",
            "default_device": "cpu",
            "Lx": 1,
            "Ly": 1,
        }
        cls.global_args = GLOBALARGS()
        cls.global_args.Lx = 1
        cls.global_args.Ly = 1
        cls.global_args.device = "cpu"
        cls.unitary = fixed_d4_to_two_spin_unitary(torch.complex128, "cpu")
        cls.model = TriangleSpinChiralPairModel(cls.config, cls.unitary)

    def test_fixed_unitary_and_factorized_operators(self):
        model = self.model
        unitary_id = id(model.split_unitary)
        self.assertLess(
            float(torch.linalg.norm(model.split_unitary.conj().T @ model.split_unitary - torch.eye(4, dtype=torch.complex128))),
            1.0e-14,
        )
        for layer in ("chiral", "antichiral"):
            Sx, Sy, Sz = model.components[layer]
            Sa, Sb, _, S1, S2, S3, _, _ = model.factorized[layer]
            ss_factorized = yastn.ncon([Sa, Sb], [[1, -1, -2], [1, -3, -4]])
            ss_exact = yastn.ncon([Sx, Sx], [[-1, -2], [-3, -4]])
            ss_exact = ss_exact + yastn.ncon([Sy, Sy], [[-1, -2], [-3, -4]])
            ss_exact = ss_exact + yastn.ncon([Sz, Sz], [[-1, -2], [-3, -4]])
            chi_factorized = yastn.ncon(
                [S1, S2, S3],
                [[-1, -4, 1], [1, -2, -5, 2], [2, -3, -6]],
            )
            chi_exact = (
                yastn.ncon([Sx, Sy, Sz], [[-1, -4], [-2, -5], [-3, -6]])
                - yastn.ncon([Sx, Sz, Sy], [[-1, -4], [-2, -5], [-3, -6]])
                + yastn.ncon([Sy, Sz, Sx], [[-1, -4], [-2, -5], [-3, -6]])
                - yastn.ncon([Sy, Sx, Sz], [[-1, -4], [-2, -5], [-3, -6]])
                + yastn.ncon([Sz, Sx, Sy], [[-1, -4], [-2, -5], [-3, -6]])
                - yastn.ncon([Sz, Sy, Sx], [[-1, -4], [-2, -5], [-3, -6]])
            )
            self.assertLess(float(yastn.linalg.norm(ss_factorized - ss_exact)), 1.0e-12)
            rel_chi = yastn.linalg.norm(chi_factorized - chi_exact) / yastn.linalg.norm(chi_exact)
            self.assertLess(float(rel_chi), 1.0e-12)
        self.assertEqual(id(model.split_unitary), unitary_id)
        self.assertEqual(model._couplings({"Jchi_chiral": 0.3, "Jchi_antichiral": -0.3})[1], {"chiral": 0.3, "antichiral": -0.3})
        with self.assertRaises(KeyError):
            model._couplings({"Jchi_chiral": 0.3})
        with self.assertRaises(ValueError):
            model._couplings({"Jchi_chiral": 0.3, "Jchi_antichiral": 0.3})

    @staticmethod
    def _run_ctm(state, global_args, chi, iterations=30):
        ctm_args = CTMARGS()
        ctm_args.chi = chi
        ctm_args.CTM_ite_nums = iterations
        ctm_args.CTM_conv_tol = 1.0e-12
        ctm_args.CTM_trun_tol = 0.0
        with torch.no_grad():
            return CTMRG_cell_iPESS(
                state.B_set,
                state.T_set,
                INITCTMARGS(),
                None,
                ctm_args,
                global_args,
            )

    @staticmethod
    def _tensor_with_template(array, template):
        tensor = yastn.zeros(config=template.config, legs=template.get_legs())
        tensor.set_block(val=array)
        return tensor

    def _product_ctm(self, single_ctm, pair_template_ctm, single_D):
        pair_ctm = OrderedDict((group, OrderedDict()) for group in ("Cset", "Tset"))
        for cell, corners in single_ctm["Cset"].items():
            pair_ctm["Cset"][cell] = OrderedDict()
            for direction, tensor in corners.items():
                array = np.asarray(tensor.to_dense().to("cpu"))
                product = np.einsum("ab,cd->acbd", array, array.conj()).reshape(
                    array.shape[0] ** 2, array.shape[1] ** 2
                )
                template = pair_template_ctm["Cset"][cell][direction]
                pair_ctm["Cset"][cell][direction] = self._tensor_with_template(product, template)
        for cell, edges in single_ctm["Tset"].items():
            pair_ctm["Tset"][cell] = OrderedDict()
            for direction, tensor in edges.items():
                array = np.asarray(tensor.to_dense().to("cpu"))
                chi_left, double_dim, chi_right = array.shape
                self.assertEqual(double_dim, single_D**2)
                unfused = array.reshape(chi_left, single_D, single_D, chi_right)
                # Reorder (Aket, Abra) x (Bket, Bbra) into
                # (Aket, Bket) x (Abra, Bbra), matching the pair double layer.
                product = np.einsum("aklb,cmnd->ackmlnbd", unfused, unfused.conj()).reshape(
                    chi_left**2, single_D**4, chi_right**2
                )
                template = pair_template_ctm["Tset"][cell][direction]
                pair_ctm["Tset"][cell][direction] = self._tensor_with_template(product, template)
        return pair_ctm

    def test_D2_single_times_conjugate_gives_D4_pair_observables(self):
        single = random_bosonic_triangle_iPESS(2, self.config, physical_dim=2, seed=17)
        pair = chiral_pair_from_single_bosonic_triangle_iPESS(single, self.unitary)
        self.assertEqual(bosonic_triangle_iPESS_bond_dimension(pair), 4)
        self.assertEqual(pair.T_set["1,1"].get_shape()[1], 4)

        single_ctm, single_dB, single_dT, _, _ = self._run_ctm(single, self.global_args, 4)
        pair_template_ctm, pair_dB, pair_dT, _, _ = self._run_ctm(
            pair, self.global_args, 16, iterations=0
        )
        pair_ctm = self._product_ctm(single_ctm, pair_template_ctm, single_D=2)
        parameters_single = {"J1": 1.0, "Jchi": 0.27}
        parameters_pair = {
            "J1": 1.0,
            "Jchi_chiral": 0.27,
            "Jchi_antichiral": -0.27,
        }
        with torch.no_grad():
            single_energy, single_obs = evaluate_triangle_spin_energy(
                parameters_single,
                single.B_set,
                single.T_set,
                single_dB,
                single_dT,
                single_ctm,
                self.config,
                self.global_args,
                return_observables=True,
            )
            pair_energy, pair_obs = self.model.evaluate(
                parameters_pair,
                pair.B_set,
                pair.T_set,
                pair_dB,
                pair_dT,
                pair_ctm,
                self.global_args,
                return_observables=True,
            )

        comparisons = {}
        for name in ("SS_x", "SS_y", "SS_diagonal"):
            comparisons[name + "_chiral"] = (pair_obs[name + "_chiral"], single_obs[name])
            comparisons[name + "_antichiral"] = (pair_obs[name + "_antichiral"], single_obs[name])
        for name in ("chi_up", "chi_down"):
            comparisons[name + "_chiral"] = (pair_obs[name + "_chiral"], single_obs[name])
            comparisons[name + "_antichiral"] = (pair_obs[name + "_antichiral"], -single_obs[name])
        comparisons["Sx_chiral"] = (pair_obs["Sx_chiral"], single_obs["Sx"])
        comparisons["Sx_antichiral"] = (pair_obs["Sx_antichiral"], single_obs["Sx"])
        comparisons["Sy_chiral"] = (pair_obs["Sy_chiral"], single_obs["Sy"])
        comparisons["Sy_antichiral"] = (pair_obs["Sy_antichiral"], -single_obs["Sy"])
        comparisons["Sz_chiral"] = (pair_obs["Sz_chiral"], single_obs["Sz"])
        comparisons["Sz_antichiral"] = (pair_obs["Sz_antichiral"], single_obs["Sz"])
        comparisons["energy_chiral"] = (pair_obs["energy_chiral"], single_energy)
        comparisons["energy_antichiral"] = (pair_obs["energy_antichiral"], single_energy)
        comparisons["energy_average_per_layer"] = (pair_energy, single_energy)
        self.assertEqual(
            set(pair_obs), set(comparisons) - {"energy_average_per_layer"}
        )

        for label, (got, expected) in comparisons.items():
            error = torch.max(torch.abs(got - expected)).item()
            with self.subTest(observable=label):
                self.assertLess(error, 2.0e-11, msg=f"{label} error={error}")

    def test_pair_energy_has_finite_gradient_through_optimizer_path(self):
        single = random_bosonic_triangle_iPESS(2, self.config, physical_dim=2, seed=23)
        pair = chiral_pair_from_single_bosonic_triangle_iPESS(single, self.unitary)
        ctm_args = CTMARGS()
        ctm_args.chi = 4
        ctm_args.CTM_ite_nums = 1
        ctm_args.CTM_conv_tol = 0.0
        energy_setting = Square_Hubbard_Energy_settings()
        energy_setting.model = self.model.model_name
        energy_setting.model_object = self.model
        grad, energy, _ = get_grad(
            {"J1": 1.0, "Jchi_chiral": 0.2, "Jchi_antichiral": -0.2},
            pair,
            ctm_args,
            energy_setting,
            self.global_args,
            self.config,
        )
        self.assertTrue(torch.isfinite(torch.as_tensor(energy.item())))
        self.assertGreater(grad.norm(), 0.0)
        self.assertTrue(torch.isfinite(torch.as_tensor(grad.norm())))

    def test_one_gradient_optimization_step(self):
        single = random_bosonic_triangle_iPESS(2, self.config, physical_dim=2, seed=31)
        pair = chiral_pair_from_single_bosonic_triangle_iPESS(single, self.unitary)
        ad_ctm_args = CTMARGS()
        ad_ctm_args.chi = 4
        ad_ctm_args.CTM_ite_nums = 1
        ad_ctm_args.CTM_conv_tol = 0.0
        ls_ctm_args = CTMARGS()
        ls_ctm_args.chi = 4
        ls_ctm_args.CTM_ite_nums = 1
        ls_ctm_args.CTM_conv_tol = 0.0
        energy_setting = Square_Hubbard_Energy_settings()
        energy_setting.model = self.model.model_name
        energy_setting.model_object = self.model
        line_search = LINESEARCH()
        line_search.method = "lbfgs"
        line_search.line_search = "backtracking"
        line_search.maxiter = 1
        line_search.ls_maxiter = 1
        line_search.step0 = 1.0e-3
        line_search.gtol = 0.0
        line_search.print_observables = False
        with tempfile.TemporaryDirectory() as directory:
            config = dict(self.config)
            config["save_file_prefix"] = os.path.join(directory, "pair_smoke")
            optimized = optimize_iPESS(
                {"J1": 1.0, "Jchi_chiral": 0.2, "Jchi_antichiral": -0.2},
                4,
                4,
                pair,
                ad_ctm_args,
                ls_ctm_args,
                energy_setting,
                self.global_args,
                config,
                line_search,
            )
        self.assertEqual(bosonic_triangle_iPESS_bond_dimension(optimized), 4)
        self.assertTrue(torch.isfinite(torch.as_tensor(optimized.norm())))


if __name__ == "__main__":
    unittest.main()
