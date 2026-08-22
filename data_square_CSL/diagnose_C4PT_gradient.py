import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy
import torch
import yastn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(r"D:\My Documents\Code\python_codes\iPEPS_Z2")

from ansatz.square_iPEPS import IPEPS_SQUARE_C4_PT, add_noise, dense_tensor_from_array, random_square_iPEPS
from config.config import CTMARGS, GLOBALARGS, INITCTMARGS
from ctmrg.bosonic_CTMRG_unitcell_iPEPS import Bosonic_CTMRG_cell_iPEPS
from model.bosonic_square_ob_iPEPS import evaluate_ob_cell_iPEPS, prl_129_177201_square_csl_parameters


def read_juraj_one_site_tensor(path):
    with open(path) as f:
        raw = json.load(f)
    site = raw["sites"][0]
    dims = tuple(site["dims"])
    A = numpy.zeros(dims, dtype=numpy.complex128)
    for entry in site["entries"]:
        fields = entry.split()
        inds = tuple(int(x) for x in fields[: len(dims)])
        A[inds] = float(fields[len(dims)]) + 1.0j * float(fields[len(dims) + 1])
    return A


def make_state_from_juraj(path, config_kwargs):
    A_juraj = read_juraj_one_site_tensor(path)
    A_square = numpy.transpose(A_juraj, (2, 3, 4, 1, 0))
    A_set = OrderedDict({"1,1": dense_tensor_from_array(A_square, config_kwargs, dual=[0, 1, 0, 1, 0])})
    A_set["1,1"] = A_set["1,1"] / torch.max(torch.abs(A_set["1,1"].to_dense()))
    return IPEPS_SQUARE_C4_PT(A_set, config_kwargs, impose_c4=True, impose_pt=True)


def yastn_diag_values(S):
    dense = S.to_dense()
    if dense.ndim == 2:
        vals = torch.diagonal(dense)
    else:
        vals = dense.reshape(-1)
    vals = vals[torch.abs(vals) > 0]
    return vals.detach().cpu()


def compress_values(T):
    data, _ = yastn.Tensor.compress_to_1d(T)
    return data


def main():
    print("pid= " + str(os.getpid()))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--juraj-state",
        default=r"D:\My Documents\Code\python_codes\Juraj\tn-torch_dev\test_csl\prl129_D3_chi40_state.json",
    )
    parser.add_argument("--chi", type=int, default=24)
    parser.add_argument("--iters", default="1,2,4,8")
    parser.add_argument("--trunc-tols", default="1e-8,1e-10")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--state-kind", choices=("saved", "random", "noisy"), default="saved")
    parser.add_argument("--bond-dim", type=int, default=3)
    parser.add_argument("--noise", type=float, default=1.0e-2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
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
    parameters = prl_129_177201_square_csl_parameters(chirality_sign=-1.0)

    orig_svd_with_truncation = yastn.linalg.svd_with_truncation
    svd_records = []

    def wrapped_svd_with_truncation(*svd_args, **svd_kwargs):
        u, s, v = orig_svd_with_truncation(*svd_args, **svd_kwargs)
        vals = torch.abs(yastn_diag_values(s))
        if vals.numel() > 0:
            svd_records.append(
                {
                    "count": int(vals.numel()),
                    "min": float(torch.min(vals)),
                    "max": float(torch.max(vals)),
                    "cond": float(torch.max(vals) / torch.clamp(torch.min(vals), min=1e-300)),
                }
            )
        return u, s, v

    yastn.linalg.svd_with_truncation = wrapped_svd_with_truncation
    try:
        print("chi,ctm_iters,trunc_tol,energy,ctm_err,grad_norm,grad_inf,svd_min,svd_cond_max,svd_calls")
        for trunc_tol in [float(x) for x in args.trunc_tols.split(",")]:
            for niter in [int(x) for x in args.iters.split(",")]:
                svd_records.clear()
                if args.state_kind == "saved":
                    state = make_state_from_juraj(args.juraj_state, config_kwargs)
                elif args.state_kind == "noisy":
                    state = make_state_from_juraj(args.juraj_state, config_kwargs)
                    state = add_noise(state, args.noise, config_kwargs)
                    state.normalize()
                else:
                    A_set = random_square_iPEPS(args.bond_dim, 2, config_kwargs)
                    state = IPEPS_SQUARE_C4_PT(A_set, config_kwargs, impose_c4=True, impose_pt=True)
                    state.normalize()
                state.require_grad(True)
                ctm_args = CTMARGS()
                ctm_args.chi = args.chi
                ctm_args.CTM_ite_nums = niter
                ctm_args.CTM_trun_tol = trunc_tol
                ctm_args.CTM_conv_tol = 0.0
                ctm_args.CTM_ite_info = False
                ctm_args.doublelayer_on_cpu = False
                ctm_args.use_sub_checkpoint = False

                CTM_cell, double_A_cell, _, ite_err = Bosonic_CTMRG_cell_iPEPS(
                    state.A_set, INITCTMARGS(), None, ctm_args, global_args
                )
                E_total, *_ = evaluate_ob_cell_iPEPS(parameters, state.A_set, double_A_cell, CTM_cell, config_kwargs, global_args)
                loss = torch.real(E_total)
                loss.backward()
                grad_data = compress_values(state.A_set["1,1"].grad())
                grad_norm = torch.linalg.vector_norm(grad_data).item()
                grad_inf = torch.max(torch.abs(grad_data)).item()
                svd_min = min((r["min"] for r in svd_records), default=float("nan"))
                svd_cond_max = max((r["cond"] for r in svd_records), default=float("nan"))
                print(
                    f"{args.chi},{niter},{trunc_tol:.1e},{loss.item():.16g},{float(ite_err):.6g},"
                    f"{grad_norm:.6g},{grad_inf:.6g},{svd_min:.6g},{svd_cond_max:.6g},{len(svd_records)}"
                )
    finally:
        yastn.linalg.svd_with_truncation = orig_svd_with_truncation


if __name__ == "__main__":
    main()
