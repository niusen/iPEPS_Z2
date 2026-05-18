import argparse
import os
import subprocess
import sys
from collections import OrderedDict

import numpy
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(r"D:\My Documents\Code\python_codes\iPEPS_Z2")

from ansatz.square_iPEPS import IPEPS_SQUARE_C4_PT, load_square_iPEPS
from config.config import CTMARGS, GLOBALARGS, INITCTMARGS
from ctmrg.bosonic_CTMRG_unitcell_iPEPS import Bosonic_CTMRG_cell_iPEPS
from model.bosonic_square_ob_iPEPS import evaluate_ob_cell_iPEPS, prl_129_177201_square_csl_parameters
from optimization.optimize_bosonic_square_iPEPS import _double_layer_for_observables


JURAJ_REPO = r"D:\My Documents\Code\python_codes\Juraj\tn-torch_dev"
JURAJ_SCRIPT = "test_csl/optim_csl_c4pt_generic_svd_from_ipepsz2.py"


def square_to_juraj(A_square):
    # iPEPS_Z2: [left, down, right, up, physical].
    # Juraj: [physical, up, left, down, right].
    return numpy.transpose(A_square, (4, 3, 0, 1, 2))


def inner_complex(a, b):
    return numpy.vdot(a.reshape(-1), b.reshape(-1))


def norm(a):
    return float(numpy.linalg.norm(a.reshape(-1)))


def make_c4pt_state_from_param(template_tensor, param, config_kwargs):
    state = IPEPS_SQUARE_C4_PT(
        OrderedDict({"1,1": template_tensor._replace(data=param)}),
        config_kwargs,
        impose_c4=True,
        impose_pt=True,
    )
    state.normalize()
    return state


def compute_ipeps_grad(args, out_npz):
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

    A_set = load_square_iPEPS(args.square_prefix, config_kwargs)
    initial_state = IPEPS_SQUARE_C4_PT(A_set, config_kwargs, impose_c4=True, impose_pt=True)
    initial_state.to_device(args.device)
    initial_state.normalize()
    template_tensor = initial_state.A_set["1,1"].copy()
    param = torch.nn.Parameter(initial_state.A_set["1,1"]._data.detach().clone())

    ctm_args = CTMARGS()
    ctm_args.chi = args.chi
    ctm_args.CTM_ite_nums = args.ctm_max_iter
    ctm_args.CTM_trun_tol = 1.0e-8
    ctm_args.CTM_conv_tol = args.ctm_conv_tol
    ctm_args.CTM_ite_info = False
    ctm_args.doublelayer_on_cpu = False
    ctm_args.use_sub_checkpoint = False

    parameters = prl_129_177201_square_csl_parameters(chirality_sign=-1.0)
    state = make_c4pt_state_from_param(template_tensor, param, config_kwargs)
    CTM_cell, double_A_cell, _ite_num, ite_err = Bosonic_CTMRG_cell_iPEPS(
        state.A_set, INITCTMARGS(), None, ctm_args, global_args
    )
    double_A_cell = _double_layer_for_observables(double_A_cell, ctm_args, global_args)
    E_total, *_ = evaluate_ob_cell_iPEPS(parameters, state.A_set, double_A_cell, CTM_cell, config_kwargs, global_args)
    loss = torch.real(E_total)
    loss.backward()

    grad_square = template_tensor._replace(data=param.grad).to_dense().detach().cpu().numpy().copy()
    param_square = template_tensor._replace(data=param.detach()).to_dense().detach().cpu().numpy().copy()
    grad = square_to_juraj(grad_square)
    param = square_to_juraj(param_square)
    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
    numpy.savez(
        out_npz,
        loss=numpy.array(float(loss.detach().cpu())),
        grad=grad,
        param=param,
        ctm_err=numpy.array(float(numpy.real(ite_err)) if not isinstance(ite_err, (list, tuple)) else numpy.nan),
    )
    print("ipeps_loss:", f"{float(loss.detach().cpu()):.16g}")
    print("ipeps_grad_norm:", f"{norm(grad):.16g}")
    print("wrote_ipeps_npz:", out_npz)


def compute_juraj_grad(args, out_npz):
    cmd = [
        sys.executable,
        os.path.join(args.juraj_repo, JURAJ_SCRIPT),
        "--instate-json",
        args.instate_json,
        "--out-prefix",
        os.path.abspath(os.path.join(args.out_dir, "juraj_grad_compare_tmp")),
        "--chi",
        str(args.chi),
        "--ctm-max-iter",
        str(args.ctm_max_iter),
        "--ctm-conv-tol",
        str(args.ctm_conv_tol),
        "--opt-max-iter",
        "0",
        "--threads",
        str(args.threads),
        "--device",
        args.device,
        "--grad-out-npz",
        out_npz,
    ]
    print("running Juraj generic-SVD grad:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=args.juraj_repo, check=True)


def compare_npz(juraj_npz, ipeps_npz):
    juraj = numpy.load(juraj_npz)
    ipeps = numpy.load(ipeps_npz)
    gj = juraj["grad"]
    gi = ipeps["grad"]
    pj = juraj["param"]
    pi = ipeps["param"]
    diff = gi - gj
    param_diff = pi - pj
    nj = norm(gj)
    ni = norm(gi)
    nd = norm(diff)
    ip = inner_complex(gj, gi)
    cosine = float(numpy.real(ip) / max(nj * ni, 1.0e-300))
    overlap_abs = float(abs(ip) / max(nj * ni, 1.0e-300))
    print("loss_juraj,", f"{float(juraj['loss']):.16g}")
    print("loss_ipeps,", f"{float(ipeps['loss']):.16g}")
    print("grad_norm_juraj,", f"{nj:.16g}")
    print("grad_norm_ipeps,", f"{ni:.16g}")
    print("grad_abs_diff,", f"{nd:.16g}")
    print("grad_rel_diff_vs_max_norm,", f"{nd / max(nj, ni, 1.0e-300):.16g}")
    print("grad_max_abs_diff,", f"{numpy.max(numpy.abs(diff)):.16g}")
    print("grad_real_cosine,", f"{cosine:.16g}")
    print("grad_abs_overlap,", f"{overlap_abs:.16g}")
    print("param_abs_diff,", f"{norm(param_diff):.16g}")
    print("param_max_abs_diff,", f"{numpy.max(numpy.abs(param_diff)):.16g}")


def main():
    print("pid= " + str(os.getpid()))
    parser = argparse.ArgumentParser()
    parser.add_argument("--instate-json", required=True)
    parser.add_argument("--square-prefix", required=True)
    parser.add_argument("--out-dir", default="data_square_CSL/grad_compare_stuck")
    parser.add_argument("--juraj-repo", default=JURAJ_REPO)
    parser.add_argument("--chi", type=int, default=40)
    parser.add_argument("--ctm-max-iter", type=int, default=80)
    parser.add_argument("--ctm-conv-tol", type=float, default=1.0e-8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-juraj", action="store_true")
    parser.add_argument("--skip-ipeps", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"chi{args.chi}_ctm{args.ctm_max_iter}"
    juraj_npz = os.path.abspath(os.path.join(args.out_dir, f"juraj_generic_svd_grad_{tag}.npz"))
    ipeps_npz = os.path.abspath(os.path.join(args.out_dir, f"ipeps_general_grad_{tag}.npz"))

    if not args.skip_juraj:
        compute_juraj_grad(args, juraj_npz)
    if not args.skip_ipeps:
        compute_ipeps_grad(args, ipeps_npz)
    compare_npz(juraj_npz, ipeps_npz)


if __name__ == "__main__":
    main()
