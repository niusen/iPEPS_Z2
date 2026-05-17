import argparse
import json
import math
import os
import subprocess
import sys
from collections import OrderedDict

import numpy
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(r"D:\My Documents\Code\python_codes\iPEPS_Z2")

from ansatz.square_iPEPS import dense_tensor_from_array, save_square_iPEPS


JURAJ_REPO = r"D:\My Documents\Code\python_codes\Juraj\tn-torch_dev"


def prl_129_177201_params():
    theta = 0.06 * math.pi
    phi = 0.14 * math.pi
    return {
        "j1": 2.0 * math.cos(theta) * math.cos(phi),
        "j2": 2.0 * math.cos(theta) * math.sin(phi),
        "lmbd": 2.0 * math.sin(theta),
    }


def read_juraj_one_site_tensor(path):
    with open(path) as f:
        raw = json.load(f)
    if len(raw["sites"]) != 1:
        raise ValueError("expected a one-site C4v Juraj state")

    site = raw["sites"][0]
    dims = tuple(site["dims"])
    A = numpy.zeros(dims, dtype=numpy.complex128)
    for entry in site["entries"]:
        fields = entry.split()
        inds = tuple(int(x) for x in fields[: len(dims)])
        A[inds] = float(fields[len(dims)]) + 1.0j * float(fields[len(dims) + 1])
    return A


def convert_juraj_state_to_square_json(juraj_state, square_prefix, device="cpu"):
    config_kwargs = {
        "backend": "torch",
        "default_dtype": "complex128",
        "default_device": device,
        "Lx": 1,
        "Ly": 1,
        "checkerboard_spin_transform": "sigmay",
    }
    # Juraj/peps-torch: p,u,l,d,r. iPEPS_Z2 square code: l,d,r,u,p.
    A_juraj = read_juraj_one_site_tensor(juraj_state)
    A_square = numpy.transpose(A_juraj, (2, 3, 4, 1, 0))
    A_set = OrderedDict({"1,1": dense_tensor_from_array(A_square, config_kwargs, dual=[0, 1, 0, 1, 0])})
    A_set["1,1"] = A_set["1,1"] / torch.max(torch.abs(A_set["1,1"].to_dense()))
    save_square_iPEPS(A_set, square_prefix, config_kwargs)


def run(cmd, cwd):
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Fast C4/PT square CSL path using Juraj's one-site C4v single-layer CTMRG backend."
    )
    parser.add_argument("--mode", choices=("eval", "opt"), default="eval")
    parser.add_argument("--juraj-repo", default=JURAJ_REPO)
    parser.add_argument("--instate", default=None, help="Juraj/peps-torch one-site state json.")
    parser.add_argument("--out-prefix", default="data_square_CSL/fast_square_C4PT_D3_chi40")
    parser.add_argument("--bond-dim", type=int, default=3)
    parser.add_argument("--chi", type=int, default=40)
    parser.add_argument("--ctm-max-iter", type=int, default=100)
    parser.add_argument("--opt-max-iter", type=int, default=100)
    parser.add_argument("--obs-freq", type=int, default=-1)
    parser.add_argument("--omp-cores", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--skip-square-export", action="store_true")
    args = parser.parse_args()

    params = prl_129_177201_params()
    py = sys.executable
    common = [
        "--bond_dim",
        str(args.bond_dim),
        "--chi",
        str(args.chi),
        "--j1",
        str(params["j1"]),
        "--j2",
        str(params["j2"]),
        "--lmbd",
        str(params["lmbd"]),
        "--GLOBALARGS_dtype",
        "complex128",
        "--omp_cores",
        str(args.omp_cores),
        "--seed",
        str(args.seed),
    ]
    if args.force_cpu:
        common.append("--force_cpu")

    if args.mode == "eval":
        if args.instate is None:
            raise ValueError("--mode eval requires --instate")
        instate_arg = args.instate
        cmd = [
            py,
            "examples/j1j2/ctmrg_j1j2lambda_c4v.py",
            "--instate",
            instate_arg,
            "--CTMARGS_ctm_max_iter",
            str(args.ctm_max_iter),
            "--CTMARGS_verbosity_ctm_convergence",
            "1",
            "--obs_freq",
            str(args.obs_freq),
            "--corrf_r",
            "1",
            "--top_n",
            "2",
            "--out_prefix",
            args.out_prefix,
        ] + common
        run(cmd, args.juraj_repo)
        juraj_state = instate_arg if os.path.isabs(instate_arg) else os.path.join(args.juraj_repo, instate_arg)
    else:
        cmd = [
            py,
            "examples/j1j2/optim_j1j2lambda_c4v.py",
            "--out_prefix",
            args.out_prefix,
            "--opt_max_iter",
            str(args.opt_max_iter),
            "--CTMARGS_verbosity_ctm_convergence",
            "1",
        ] + common
        if args.instate is not None:
            cmd += ["--instate", args.instate]
        run(cmd, args.juraj_repo)
        juraj_state = os.path.join(args.juraj_repo, args.out_prefix + "_state.json")

    if not args.skip_square_export:
        square_prefix = os.path.join(args.juraj_repo, args.out_prefix + "_iPEPS_Z2")
        convert_juraj_state_to_square_json(juraj_state, square_prefix, device="cpu")
        print("exported_iPEPS_Z2_state:", square_prefix + ".json")


if __name__ == "__main__":
    main()
