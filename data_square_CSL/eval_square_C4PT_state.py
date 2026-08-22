import argparse
import os
import sys

import numpy
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(r"D:\My Documents\Code\python_codes\iPEPS_Z2")

from ansatz.square_iPEPS import IPEPS_SQUARE_C4_PT, load_square_iPEPS
from config.config import CTMARGS, GLOBALARGS, INITCTMARGS
from ctmrg.bosonic_CTMRG_unitcell_iPEPS import Bosonic_CTMRG_cell_iPEPS
from model.bosonic_square_ob_iPEPS import (
    evaluate_ob_cell_iPEPS,
    prl_129_177201_square_csl_parameters,
)


def to_real_float(x):
    if hasattr(x, "item"):
        x = x.item()
    return float(numpy.real(x))


def main():
    print("pid= " + str(os.getpid()))
    parser = argparse.ArgumentParser()
    parser.add_argument("--instate-prefix", required=True)
    parser.add_argument("--chi", type=int, default=40)
    parser.add_argument("--ctm-iters", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chirality-sign", type=float, default=-1.0)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

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

    A_set = load_square_iPEPS(args.instate_prefix, config_kwargs)
    state = IPEPS_SQUARE_C4_PT(A_set, config_kwargs, impose_c4=True, impose_pt=True)
    state.require_grad(False)
    state.to_device(args.device)
    state.normalize()

    ctm_args = CTMARGS()
    ctm_args.CTM_ite_info = True
    ctm_args.chi = args.chi
    ctm_args.CTM_ite_nums = args.ctm_iters
    ctm_args.CTM_trun_tol = 1.0e-8
    ctm_args.CTM_conv_tol = 1.0e-6
    ctm_args.doublelayer_on_cpu = args.device.startswith("cuda")
    ctm_args.use_sub_checkpoint = False

    parameters = prl_129_177201_square_csl_parameters(chirality_sign=args.chirality_sign)
    CTM_cell, double_A_cell, ite_num, ite_err = Bosonic_CTMRG_cell_iPEPS(
        state.A_set, INITCTMARGS(), None, ctm_args, global_args
    )
    (
        E_total,
        SS_x_set,
        SS_y_set,
        SS_diag_a_set,
        SS_diag_b_set,
        triangle_no_LU_set,
        triangle_no_LD_set,
        triangle_no_RD_set,
        triangle_no_RU_set,
    ) = evaluate_ob_cell_iPEPS(parameters, state.A_set, double_A_cell, CTM_cell, config_kwargs, global_args)

    print("instate_prefix:", args.instate_prefix)
    print("chi:", args.chi)
    print("ctm_ite_num:", ite_num)
    print("ctm_ite_err:", ite_err)
    print("energy:", "{:.16g}".format(to_real_float(E_total)))
    print("SS_x:", SS_x_set)
    print("SS_y:", SS_y_set)
    print("SS_diag_a:", SS_diag_a_set)
    print("SS_diag_b:", SS_diag_b_set)
    print("chirality_no_LU:", triangle_no_LU_set)
    print("chirality_no_LD:", triangle_no_LD_set)
    print("chirality_no_RD:", triangle_no_RD_set)
    print("chirality_no_RU:", triangle_no_RU_set)


if __name__ == "__main__":
    main()
