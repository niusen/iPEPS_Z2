import argparse
import os
import sys
import time
from collections import OrderedDict

import numpy
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IPEPS_REPO = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, IPEPS_REPO)

from ansatz.square_iPEPS import IPEPS_SQUARE_C4_PT, load_square_iPEPS, save_square_iPEPS
from config.config import CTMARGS, GLOBALARGS, INITCTMARGS
from ctmrg.bosonic_CTMRG_unitcell_iPEPS import Bosonic_CTMRG_cell_iPEPS
from model.bosonic_square_ob_iPEPS import evaluate_ob_cell_iPEPS, prl_129_177201_square_csl_parameters
from optimization.lbfgs_modified import LBFGS_MOD
from optimization.optimize_bosonic_square_iPEPS import _double_layer_for_observables


def to_float(x):
    if hasattr(x, "item"):
        x = x.item()
    return float(numpy.real(x))


def make_state_from_param(template_tensor, param, config_kwargs):
    A = template_tensor._replace(data=param)
    state = IPEPS_SQUARE_C4_PT(OrderedDict({"1,1": A}), config_kwargs, impose_c4=True, impose_pt=True)
    state.normalize()
    return state


def eval_energy(parameters, state, ctm_args, global_args, config_kwargs):
    CTM_cell, double_A_cell, ite_num, ite_err = Bosonic_CTMRG_cell_iPEPS(
        state.A_set, INITCTMARGS(), None, ctm_args, global_args
    )
    double_A_cell = _double_layer_for_observables(double_A_cell, ctm_args, global_args)
    energy, *_ = evaluate_ob_cell_iPEPS(parameters, state.A_set, double_A_cell, CTM_cell, config_kwargs, global_args)
    return torch.real(energy), ite_num, ite_err


def save_raw_snapshot(out_dir, tag, param, grad, loss_value, post_energy_value, extra):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, tag + ".npz")
    arrays = {
        "raw_param": param.detach().cpu().numpy().copy(),
        "loss": numpy.array(loss_value),
        "post_energy": numpy.array(post_energy_value),
    }
    if grad is not None:
        arrays["raw_grad"] = grad.detach().cpu().numpy().copy()
        arrays["raw_grad_norm"] = numpy.array(grad.norm().item())
    for key, value in extra.items():
        arrays[key] = numpy.array(value)
    numpy.savez(path, **arrays)
    print("saved_raw_snapshot:", path)


def main():
    print("pid= " + str(os.getpid()))
    parser = argparse.ArgumentParser()
    parser.add_argument("--instate-prefix", default="../ipepsz2_random_C4PT_init_D3_seed0")
    parser.add_argument("--out-dir", default="capture_470_seed0")
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--chi", type=int, default=40)
    parser.add_argument("--maxiter", type=int, default=4)
    parser.add_argument("--ctm-iters", type=int, default=80)
    parser.add_argument("--ctm-conv-tol", type=float, default=1.0e-8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--history-size", type=int, default=6)
    parser.add_argument("--line-search", choices=("backtracking", "strong_wolfe", "none"), default="backtracking")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

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
    ctm_args.CTM_ite_nums = args.ctm_iters
    ctm_args.CTM_trun_tol = 1.0e-8
    ctm_args.CTM_conv_tol = args.ctm_conv_tol
    ctm_args.CTM_ite_info = False
    ctm_args.doublelayer_on_cpu = False
    ctm_args.use_sub_checkpoint = False

    parameters = prl_129_177201_square_csl_parameters(chirality_sign=-1.0)
    A_set = load_square_iPEPS(args.instate_prefix, config_kwargs)
    initial_state = IPEPS_SQUARE_C4_PT(A_set, config_kwargs, impose_c4=True, impose_pt=True)
    initial_state.to_device(args.device)
    initial_state.normalize()
    template_tensor = initial_state.A_set["1,1"].copy()
    param = torch.nn.Parameter(initial_state.A_set["1,1"]._data.detach().clone())

    line_search = None if args.line_search == "none" else args.line_search
    optimizer = LBFGS_MOD(
        [param],
        max_iter=1,
        lr=args.lr,
        tolerance_grad=1.0e-5,
        tolerance_change=1.0e-9,
        history_size=args.history_size,
        line_search_fn=line_search,
        line_search_eps=1.0e-8,
    )

    last_loss_value = float("nan")
    last_ctm_steps = -1
    last_ctm_err = float("nan")
    current_step = 0
    closure_call_in_step = 0

    def closure(linesearching=False):
        nonlocal last_loss_value, last_ctm_steps, last_ctm_err, closure_call_in_step
        optimizer.zero_grad()
        state = make_state_from_param(template_tensor, param, config_kwargs)
        loss, ite_num, ite_err = eval_energy(parameters, state, ctm_args, global_args, config_kwargs)
        loss.backward()
        last_loss_value = to_float(loss)
        last_ctm_steps = int(ite_num)
        last_ctm_err = to_float(ite_err)
        closure_call_in_step += 1
        save_raw_snapshot(
            args.out_dir,
            f"step{current_step}_closure{closure_call_in_step}_raw",
            param,
            param.grad,
            last_loss_value,
            float("nan"),
            {
                "step": current_step,
                "closure_call": closure_call_in_step,
                "ctm_steps": last_ctm_steps,
                "ctm_err": last_ctm_err,
                "linesearching": bool(linesearching),
            },
        )
        print("E=" + "{:.16g}".format(last_loss_value))
        return loss

    @torch.no_grad()
    def closure_linesearch(linesearching=True):
        state = make_state_from_param(template_tensor, param, config_kwargs)
        loss, _ite_num, _ite_err = eval_energy(parameters, state, ctm_args, global_args, config_kwargs)
        print("E=" + "{:.16g}".format(to_float(loss)))
        return loss

    start = time.perf_counter()
    print(
        f"capture raw C4/PT LBFGS: chi={args.chi}, maxiter={args.maxiter}, "
        f"ctm_iters={args.ctm_iters}, ctm_conv_tol={args.ctm_conv_tol}, "
        f"seed={args.seed}, lr={args.lr}, history_size={args.history_size}, "
        f"line_search={args.line_search}, instate_prefix={args.instate_prefix}, out_dir={args.out_dir}"
    )

    save_raw_snapshot(
        args.out_dir,
        "step0_initial_raw",
        param,
        None,
        float("nan"),
        float("nan"),
        {"ctm_steps": -1, "ctm_err": numpy.nan, "step": 0},
    )

    for step in range(1, args.maxiter + 1):
        current_step = step
        closure_call_in_step = 0
        step_start = time.perf_counter()
        try:
            loss_tensor = optimizer.step_2c(closure, closure_linesearch)
            step_status = "ok"
            loss_value = to_float(loss_tensor)
        except RuntimeError as exc:
            step_status = "runtime_error:" + str(exc)
            loss_value = last_loss_value

        with torch.no_grad():
            post_state = make_state_from_param(template_tensor, param, config_kwargs)
            post_energy, post_ctm_steps, post_ctm_err = eval_energy(parameters, post_state, ctm_args, global_args, config_kwargs)
            post_energy_value = to_float(post_energy)
            save_square_iPEPS(post_state.A_set, os.path.join(args.out_dir, f"step{step}_projected_state"), config_kwargs)

        grad_norm = param.grad.norm().item() if param.grad is not None else float("nan")
        elapsed = time.perf_counter() - start
        step_s = time.perf_counter() - step_start
        print(
            f"step,{step},status,{step_status},loss,{loss_value:.16g},"
            f"post_energy,{post_energy_value:.16g},raw_grad_norm,{grad_norm:.16g},"
            f"closure_ctm_steps,{last_ctm_steps},closure_ctm_err,{last_ctm_err:.16g},"
            f"post_ctm_steps,{post_ctm_steps},post_ctm_err,{to_float(post_ctm_err):.16g},"
            f"step_s,{step_s:.6g},elapsed_s,{elapsed:.6g}"
        )
        save_raw_snapshot(
            args.out_dir,
            f"step{step}_after_update_raw",
            param,
            param.grad,
            loss_value,
            post_energy_value,
            {
                "step": step,
                "closure_ctm_steps": last_ctm_steps,
                "closure_ctm_err": last_ctm_err,
                "post_ctm_steps": post_ctm_steps,
                "post_ctm_err": to_float(post_ctm_err),
            },
        )
        if step_status != "ok":
            break


if __name__ == "__main__":
    main()
