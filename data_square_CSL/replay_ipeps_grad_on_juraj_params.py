import argparse
import os
import sys
from collections import OrderedDict

import numpy
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(r"D:\My Documents\Code\python_codes\iPEPS_Z2")

from ansatz.square_iPEPS import IPEPS_SQUARE, IPEPS_SQUARE_C4_PT, dense_tensor_from_array
from config.config import CTMARGS, GLOBALARGS, INITCTMARGS
from ctmrg.bosonic_CTMRG_unitcell_iPEPS import Bosonic_CTMRG_cell_iPEPS
from model.bosonic_square_ob_iPEPS import (
    evaluate_ob_cell_iPEPS,
    prl_129_177201_square_csl_parameters,
)
from optimization.optimize_bosonic_square_iPEPS import _double_layer_for_observables


def juraj_to_square(A_juraj):
    # Juraj/peps-torch: [p,u,l,d,r]. iPEPS_Z2 square: [l,d,r,u,p].
    return numpy.transpose(A_juraj, (2, 3, 4, 1, 0))


def square_to_juraj(A_square):
    return numpy.transpose(A_square, (4, 3, 0, 1, 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--juraj-trace", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--chi", type=int, default=40)
    parser.add_argument("--ctm-max-iter", type=int, default=80)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    trace = numpy.load(args.juraj_trace)
    juraj_params = trace["params"]
    juraj_grads = trace["grads"]
    juraj_losses = trace["losses"]

    config_kwargs = {
        "backend": "torch",
        "default_dtype": "complex128",
        "default_device": args.device,
        "Lx": 1,
        "Ly": 1,
        "checkerboard_spin_transform": "sigmay",
    }
    global_args = GLOBALARGS()
    global_args.Lx = 1
    global_args.Ly = 1
    global_args.device = args.device

    ctm_args = CTMARGS()
    ctm_args.chi = args.chi
    ctm_args.CTM_ite_nums = args.ctm_max_iter
    ctm_args.CTM_trun_tol = 1.0e-8
    ctm_args.CTM_conv_tol = 1.0e-8
    ctm_args.CTM_ite_info = False
    ctm_args.doublelayer_on_cpu = False
    ctm_args.use_sub_checkpoint = False

    energy_setting = prl_129_177201_square_csl_parameters(chirality_sign=-1.0)

    losses = []
    grads = []
    print("step,juraj_loss,ipeps_replay_loss,grad_norm_juraj,grad_norm_ipeps,abs_diff,rel_diff,max_abs_diff")
    for step, A_juraj in enumerate(juraj_params):
        A_square = juraj_to_square(A_juraj)
        A_raw = dense_tensor_from_array(A_square, config_kwargs, dual=[0, 1, 0, 1, 0])
        raw_state = IPEPS_SQUARE(OrderedDict({"1,1": A_raw}), config_kwargs)
        raw_state.require_grad(True)

        eval_state = IPEPS_SQUARE_C4_PT(OrderedDict({"1,1": raw_state.A_set["1,1"]}), config_kwargs)
        eval_state.normalize()
        CTM_cell, double_A_cell, _, _ = Bosonic_CTMRG_cell_iPEPS(
            eval_state.A_set, INITCTMARGS(), None, ctm_args, global_args
        )
        double_A_cell = _double_layer_for_observables(double_A_cell, ctm_args, global_args)
        E_total, *_ = evaluate_ob_cell_iPEPS(
            energy_setting, eval_state.A_set, double_A_cell, CTM_cell, config_kwargs, global_args
        )
        E = torch.real(E_total)
        E.backward()
        grad_square = raw_state.A_set["1,1"].grad().to_dense().detach().cpu().numpy()
        grad_juraj_order = square_to_juraj(grad_square)

        losses.append(float(E.detach().cpu()))
        grads.append(grad_juraj_order.copy())

        gj = juraj_grads[step]
        gi = grad_juraj_order
        diff = gi - gj
        nj = numpy.linalg.norm(gj.reshape(-1))
        ni = numpy.linalg.norm(gi.reshape(-1))
        nd = numpy.linalg.norm(diff.reshape(-1))
        denom = max(nj, ni, 1.0e-300)
        print(
            f"{step + 1},"
            f"{juraj_losses[step]:.16g},"
            f"{losses[-1]:.16g},"
            f"{nj:.16g},"
            f"{ni:.16g},"
            f"{nd:.16g},"
            f"{nd / denom:.16g},"
            f"{numpy.max(numpy.abs(diff)):.16g}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)), exist_ok=True)
    numpy.savez(args.out_npz, losses=numpy.array(losses), grads=numpy.array(grads))


if __name__ == "__main__":
    main()
