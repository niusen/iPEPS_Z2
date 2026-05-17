import argparse
import os
import subprocess
import sys
import time
from collections import OrderedDict

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
from optimization.optimize_bosonic_square_iPEPS import _double_layer_for_observables
from optimization.lbfgs_modified import LBFGS_MOD


JURAJ_REPO = r"D:\My Documents\Code\python_codes\Juraj\tn-torch_dev"

def dense_square_to_juraj(A_square):
    # iPEPS_Z2 stores [l,d,r,u,p], Juraj/peps-torch stores [p,u,l,d,r].
    return numpy.transpose(A_square, (4, 3, 0, 1, 2))


def run_juraj_worker(args, params, out_npz):
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "juraj_grad_trace_worker.py")
    cmd = [
        sys.executable,
        script,
        "--juraj-repo",
        args.juraj_repo,
        "--instate",
        args.juraj_instate,
        "--out-npz",
        out_npz,
        "--bond-dim",
        str(args.D),
        "--chi",
        str(args.chi),
        "--ctm-max-iter",
        str(args.ctm_max_iter),
        "--opt-max-iter",
        str(args.steps),
        "--j1",
        str(params["j1"]),
        "--j2",
        str(params["j2"]),
        "--lmbd",
        str(params["lmbd"]),
        "--seed",
        str(args.seed),
        "--threads",
        str(args.threads),
        "--lr",
        str(args.lr),
        "--history-size",
        str(args.history_size),
        "--line-search",
        args.line_search,
    ]
    print("running Juraj worker:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=args.juraj_repo, check=True)


def run_ipeps_trace(args, params, out_npz):
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

    A_set = load_square_iPEPS(args.square_prefix, config_kwargs)
    state = IPEPS_SQUARE_C4_PT(A_set, config_kwargs, impose_c4=True, impose_pt=True)
    state.to_device(args.device)
    state.require_grad(True)
    param = state.A_set["1,1"]._data
    param.requires_grad_(True)

    ctm_args = CTMARGS()
    ctm_args.chi = args.chi
    ctm_args.CTM_ite_nums = args.ctm_max_iter
    ctm_args.CTM_trun_tol = 1.0e-8
    ctm_args.CTM_conv_tol = 1.0e-8
    ctm_args.CTM_ite_info = False
    ctm_args.doublelayer_on_cpu = False
    ctm_args.use_sub_checkpoint = False

    energy_setting = prl_129_177201_square_csl_parameters(chirality_sign=-1.0)

    optimizer = LBFGS_MOD(
        [param],
        max_iter=1,
        lr=args.lr,
        tolerance_grad=1.0e-5,
        tolerance_change=1.0e-9,
        history_size=args.history_size,
        line_search_fn=args.line_search,
        line_search_eps=1.0e-8,
    )

    traces = {"loss": [], "grad": [], "param": [], "ls_loss": [], "step_time": []}

    def make_eval_state():
        # Project and normalize inside the loss, matching Juraj's to_ipeps_c4v(..., normalize=True)
        # pattern as closely as the iPEPS_Z2 parameterization allows.
        eval_A_set = OrderedDict({"1,1": state.A_set["1,1"]})
        eval_state = IPEPS_SQUARE_C4_PT(eval_A_set, config_kwargs, impose_c4=True, impose_pt=True)
        eval_state.normalize()
        return eval_state

    def loss_for_state():
        eval_state = make_eval_state()
        CTM_cell, double_A_cell, _, _ = Bosonic_CTMRG_cell_iPEPS(
            eval_state.A_set, INITCTMARGS(), None, ctm_args, global_args
        )
        double_A_cell = _double_layer_for_observables(double_A_cell, ctm_args, global_args)
        E_total, *_ = evaluate_ob_cell_iPEPS(
            energy_setting, eval_state.A_set, double_A_cell, CTM_cell, config_kwargs, global_args
        )
        return torch.real(E_total)

    def closure(linesearching=False):
        optimizer.zero_grad()
        loss = loss_for_state()
        loss.backward()
        if not linesearching:
            grad_tensor = state.A_set["1,1"].grad().to_dense().detach().cpu().numpy().copy()
            param_tensor = state.A_set["1,1"].to_dense().detach().cpu().numpy().copy()
            traces["loss"].append(float(loss.detach().cpu()))
            traces["grad"].append(dense_square_to_juraj(grad_tensor))
            traces["param"].append(dense_square_to_juraj(param_tensor))
        return loss

    @torch.no_grad()
    def closure_linesearch(linesearching=True):
        loss = loss_for_state()
        traces["ls_loss"].append(float(loss.detach().cpu()))
        return loss

    for _ in range(args.steps):
        step_start = time.perf_counter()
        optimizer.step_2c(closure, closure_linesearch)
        traces["step_time"].append(time.perf_counter() - step_start)

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    numpy.savez(
        out_npz,
        losses=numpy.array(traces["loss"]),
        grads=numpy.array(traces["grad"]),
        params=numpy.array(traces["param"]),
        ls_losses=numpy.array(traces["ls_loss"]),
        step_times=numpy.array(traces["step_time"]),
    )
    print("wrote", out_npz)


def compare_traces(juraj_npz, ipeps_npz):
    juraj = numpy.load(juraj_npz)
    ipeps = numpy.load(ipeps_npz)
    n = min(len(juraj["grads"]), len(ipeps["grads"]))
    has_times = "step_times" in juraj.files and "step_times" in ipeps.files
    header = "step,juraj_loss,ipeps_loss,grad_norm_juraj,grad_norm_ipeps,abs_diff,rel_diff,max_abs_diff"
    if has_times:
        header += ",juraj_step_s,ipeps_step_s,speedup_juraj_over_ipeps"
    print(header)
    for i in range(n):
        gj = juraj["grads"][i]
        gi = ipeps["grads"][i]
        diff = gi - gj
        nj = numpy.linalg.norm(gj.reshape(-1))
        ni = numpy.linalg.norm(gi.reshape(-1))
        nd = numpy.linalg.norm(diff.reshape(-1))
        denom = max(nj, ni, 1.0e-300)
        line = (
            f"{i + 1},"
            f"{juraj['losses'][i]:.16g},"
            f"{ipeps['losses'][i]:.16g},"
            f"{nj:.16g},"
            f"{ni:.16g},"
            f"{nd:.16g},"
            f"{nd / denom:.16g},"
            f"{numpy.max(numpy.abs(diff)):.16g}"
        )
        if has_times:
            tj = float(juraj["step_times"][i])
            ti = float(ipeps["step_times"][i])
            line += f",{tj:.6g},{ti:.6g},{ti / tj:.6g}"
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--juraj-repo", default=JURAJ_REPO)
    parser.add_argument("--juraj-instate", required=True)
    parser.add_argument("--square-prefix", required=True)
    parser.add_argument("--out-dir", default="data_square_CSL/grad_trace_compare")
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--chi", type=int, default=24)
    parser.add_argument("--ctm-max-iter", type=int, default=60)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--history-size", type=int, default=6)
    parser.add_argument("--line-search", default="backtracking")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-juraj", action="store_true")
    parser.add_argument("--skip-ipeps", action="store_true")
    args = parser.parse_args()

    params = {
        "j1": 1.7776001555035785,
        "j2": 0.8364751394573281,
        "lmbd": 0.3747626291714492,
    }
    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"D{args.D}_chi{args.chi}_seed{args.seed}_steps{args.steps}"
    juraj_npz = os.path.join(args.out_dir, f"juraj_{tag}.npz")
    ipeps_npz = os.path.join(args.out_dir, f"ipeps_{tag}.npz")

    if not args.skip_juraj:
        run_juraj_worker(args, params, juraj_npz)
    if not args.skip_ipeps:
        run_ipeps_trace(args, params, ipeps_npz)
    compare_traces(juraj_npz, ipeps_npz)


if __name__ == "__main__":
    main()
