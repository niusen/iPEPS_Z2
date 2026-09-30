import unittest

import torch

from ansatz.bosonic_triangle_iPESS import random_bosonic_triangle_iPESS
from ansatz.triangle_iPESS import CTM_to_device, Cell_to_device
from config.config import CTMARGS, GLOBALARGS, INITCTMARGS
from ctmrg.CTMRG_unitcell_iPESS import CTMRG_cell_iPESS
from model.bosonic_spin_ob_iPESS import evaluate_triangle_spin_energy


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for device migration")
class TriangleSpinObservableDeviceTests(unittest.TestCase):
    def test_gpu_ctm_then_cpu_observables(self):
        config = {
            "backend": "torch",
            "default_dtype": "complex128",
            "default_device": "cuda:0",
            "Lx": 1,
            "Ly": 1,
        }
        global_args = GLOBALARGS()
        global_args.Lx = 1
        global_args.Ly = 1
        global_args.device = "cuda:0"
        state = random_bosonic_triangle_iPESS(2, config, seed=123)
        state.require_grad(False)

        ctm_args = CTMARGS()
        ctm_args.chi = 4
        ctm_args.CTM_ite_nums = 1
        ctm_args.doublelayer_on_cpu = False
        ctm_args.use_sub_checkpoint = False
        with torch.no_grad():
            CTM, double_B, double_T, _, _ = CTMRG_cell_iPESS(
                state.B_set,
                state.T_set,
                INITCTMARGS(),
                None,
                ctm_args,
                global_args,
            )
            CTM = CTM_to_device(CTM, "cpu", global_args)
            double_B = Cell_to_device(double_B, "cpu", global_args)
            double_T = Cell_to_device(double_T, "cpu", global_args)
            state.to_device("cpu")
            energy, observables = evaluate_triangle_spin_energy(
                {"J1": 1.0, "Jchi": 0.4},
                state.B_set,
                state.T_set,
                double_B,
                double_T,
                CTM,
                config,
                global_args,
                return_observables=True,
            )

        self.assertEqual(energy.device.type, "cpu")
        self.assertTrue(torch.isfinite(energy))
        for value in observables.values():
            self.assertEqual(value.device.type, "cpu")
            self.assertTrue(torch.all(torch.isfinite(value)))


if __name__ == "__main__":
    unittest.main()
