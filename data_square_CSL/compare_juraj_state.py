import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(r"D:\My Documents\Code\python_codes\iPEPS_Z2")

from ansatz.square_iPEPS import dense_tensor_from_array
from config.config import CTMARGS, GLOBALARGS, INITCTMARGS
from ctmrg.bosonic_CTMRG_unitcell_iPEPS import Bosonic_CTMRG_cell_iPEPS
from model.bosonic_square_ob_iPEPS import (
    evaluate_ob_cell_iPEPS,
    prl_129_177201_square_csl_parameters,
)


def read_juraj_one_site_tensor(path):
    with open(path) as f:
        raw = json.load(f)
    if len(raw["sites"]) != 1:
        raise ValueError("expected a one-site Juraj state")

    site = raw["sites"][0]
    dims = tuple(site["dims"])
    A = numpy.zeros(dims, dtype=numpy.complex128)
    for entry in site["entries"]:
        fields = entry.split()
        inds = tuple(int(x) for x in fields[: len(dims)])
        A[inds] = float(fields[len(dims)]) + 1.0j * float(fields[len(dims) + 1])
    return A


def to_real_float(x):
    if hasattr(x, "item"):
        x = x.item()
    return float(numpy.real(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--juraj-state",
        default=r"D:\My Documents\Code\python_codes\Juraj\tn-torch_dev\test_csl\prl129_D3_chi40_state.json",
    )
    parser.add_argument("--chi", type=int, default=40)
    parser.add_argument("--ctm-iters", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkerboard-spin-transform", default="sigmay")
    parser.add_argument("--chirality-sign", type=float, default=-1.0)
    args = parser.parse_args()

    torch.set_num_threads(4)

    config_kwargs = {
        "backend": "torch",
        "default_dtype": "complex128",
        "default_device": args.device,
        "Lx": 1,
        "Ly": 1,
        "checkerboard_spin_transform": args.checkerboard_spin_transform,
    }
    global_args = GLOBALARGS()
    global_args.Lx = 1
    global_args.Ly = 1
    global_args.device = args.device

    # Juraj/peps-torch stores C4V sites as p,u,l,d,r. The iPEPS_Z2 square code
    # uses l,d,r,u,p.
    A_juraj = read_juraj_one_site_tensor(args.juraj_state)
    A_square = numpy.transpose(A_juraj, (2, 3, 4, 1, 0))
    A_set = OrderedDict({"1,1": dense_tensor_from_array(A_square, config_kwargs, dual=[0, 1, 0, 1, 0])})
    A_set["1,1"] = A_set["1,1"] / torch.max(torch.abs(A_set["1,1"].to_dense()))

    ctm_args = CTMARGS()
    ctm_args.CTM_ite_info = True
    ctm_args.chi = args.chi
    ctm_args.CTM_ite_nums = args.ctm_iters
    ctm_args.CTM_trun_tol = 1.0e-8
    ctm_args.doublelayer_on_cpu = args.device.startswith("cuda")
    ctm_args.use_sub_checkpoint = False

    parameters = prl_129_177201_square_csl_parameters(chirality_sign=args.chirality_sign)
    CTM_cell, double_A_cell, ite_num, ite_err = Bosonic_CTMRG_cell_iPEPS(
        A_set, INITCTMARGS(), None, ctm_args, global_args
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
    ) = evaluate_ob_cell_iPEPS(parameters, A_set, double_A_cell, CTM_cell, config_kwargs, global_args)

    print("juraj_state:", args.juraj_state)
    print("chi:", args.chi)
    print("chirality_sign:", args.chirality_sign)
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
