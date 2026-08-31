import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import torch

from ansatz.bosonic_triangle_iPESS import random_bosonic_triangle_iPESS
from config.config import CTMARGS, GLOBALARGS, INITCTMARGS
from ctmrg.CTMRG_unitcell_iPESS import CTMRG_cell_iPESS
from entanglement.generic_eh_momentum_stream import (
    apply_expEH_once,
    get_EH_spec_Ttensor_momentum,
)
from entanglement.triangle_ipess_adapter import (
    build_dense_ipeps_and_env,
    conversion_double_layer_relative_error,
    ipess_to_ipeps_yastn,
)


class TriangleEntanglementSpectrumTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "backend": "torch",
            "default_dtype": "complex128",
            "default_device": "cpu",
            "Lx": 2,
            "Ly": 1,
        }
        self.global_args = GLOBALARGS()
        self.global_args.Lx = 2
        self.global_args.Ly = 1
        self.global_args.device = "cpu"
        self.state = random_bosonic_triangle_iPESS(2, self.config, seed=9)

    def test_ipess_to_ipeps_double_layer_is_exact(self):
        A_set = ipess_to_ipeps_yastn(
            self.state.B_set, self.state.T_set, self.global_args
        )
        errors = conversion_double_layer_relative_error(
            self.state.B_set, self.state.T_set, A_set
        )
        self.assertLess(max(errors.values()), 1.0e-12)

    def test_streamed_chi_matches_dense_chi_for_2x1_cell(self):
        ctm_args = CTMARGS()
        ctm_args.chi = 4
        ctm_args.CTM_ite_nums = 1
        ctm_args.CTM_conv_tol = 0.0
        with torch.no_grad():
            CTM_cell, _, _, _, _ = CTMRG_cell_iPESS(
                self.state.B_set,
                self.state.T_set,
                INITCTMARGS(),
                None,
                ctm_args,
                self.global_args,
            )
            _, state, env = build_dense_ipeps_and_env(
                self.state.B_set,
                self.state.T_set,
                CTM_cell,
                self.global_args,
            )
            vector = np.arange(4, dtype=np.float64) + 1j * np.arange(4, 8)
            streamed = apply_expEH_once(
                vector, 2, (0, 0), (1, 0), state, env, stream_chi=True
            )
            dense = apply_expEH_once(
                vector, 2, (0, 0), (1, 0), state, env, stream_chi=False
            )
        relative_error = torch.linalg.norm(streamed - dense) / torch.linalg.norm(dense)
        self.assertLess(float(relative_error), 1.0e-12)

    def test_each_eigensolver_matvec_is_reported(self):
        class FakeState:
            lX = 1
            lY = 1
            device = "cpu"

            @staticmethod
            def vertexToSite(_coord):
                return (0, 0)

            @staticmethod
            def site(_coord):
                return torch.zeros((2, 2, 2, 2, 2), dtype=torch.complex128)

        class FakeEnv:
            chi = 1
            dtype = torch.complex128
            device = "cpu"

        def identity_matvec(v0, *_args, **_kwargs):
            return torch.as_tensor(v0, dtype=torch.complex128)

        output = io.StringIO()
        with mock.patch(
            "entanglement.generic_eh_momentum_stream.apply_expEH_once",
            side_effect=identity_matvec,
        ), redirect_stdout(output):
            get_EH_spec_Ttensor_momentum(
                1,
                2,
                (0, 0),
                (1, 0),
                FakeState(),
                FakeEnv(),
                full_diag_threshold=4,
                stream_chi=True,
                print_each_matvec=True,
            )

        progress = [
            line for line in output.getvalue().splitlines()
            if line.startswith("EH_MATVEC_DONE")
        ]
        self.assertEqual(len(progress), 4)
        self.assertIn("L=2 count=4 stream_chi=True", progress[-1])


if __name__ == "__main__":
    unittest.main()
