import argparse
import os
import sys
from collections import OrderedDict

import numpy
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IPEPS_REPO = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, IPEPS_REPO)

from ansatz.square_iPEPS import IPEPS_SQUARE_C4_PT, load_square_iPEPS
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
    state = IPEPS_SQUARE_C4_PT(
        OrderedDict({"1,1": template_tensor._replace(data=param)}),
        config_kwargs,
        impose_c4=True,
        impose_pt=True,
    )
    state.normalize()
    return state


def eval_loss(parameters, state, ctm_args, global_args, config_kwargs):
    CTM_cell, double_A_cell, ite_num, ite_err = Bosonic_CTMRG_cell_iPEPS(
        state.A_set, INITCTMARGS(), None, ctm_args, global_args
    )
    double_A_cell = _double_layer_for_observables(double_A_cell, ctm_args, global_args)
    loss, *_ = evaluate_ob_cell_iPEPS(parameters, state.A_set, double_A_cell, CTM_cell, config_kwargs, global_args)
    return torch.real(loss), ite_num, ite_err


def compute_grad(label, optimizer, template_tensor, param, parameters, ctm_args, global_args, config_kwargs):
    optimizer.zero_grad()
    state = make_state_from_param(template_tensor, param, config_kwargs)
    loss, ite_num, ite_err = eval_loss(parameters, state, ctm_args, global_args, config_kwargs)
    loss.backward()
    grad_norm = param.grad.norm().item()
    max_abs = param.grad.abs().max().item()
    print(
        f"{label},loss,{to_float(loss):.16g},grad_norm,{grad_norm:.16g},"
        f"max_abs_grad,{max_abs:.16g},ctm_steps,{ite_num},ctm_err,{to_float(ite_err):.16g}"
    )
    return loss


def main():
    print("pid= " + str(os.getpid()))
    parser = argparse.ArgumentParser()
    parser.add_argument("--instate-prefix", default="../ipepsz2_random_C4PT_init_D3_seed0")
    parser.add_argument("--chi", type=int, default=40)
    parser.add_argument("--ctm-iters", type=int, default=80)
    parser.add_argument("--ctm-conv-tol", type=float, default=1.0e-8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--history-size", type=int, default=6)
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

    ctm_args = CTMARGS()
    ctm_args.chi = args.chi
    ctm_args.CTM_ite_nums = args.ctm_iters
    ctm_args.CTM_trun_tol = 1.0e-8
    ctm_args.CTM_conv_tol = args.ctm_conv_tol
    ctm_args.CTM_ite_info = False
    ctm_args.doublelayer_on_cpu = False
    ctm_args.use_sub_checkpoint = False

    parameters = prl_129_177201_square_csl_parameters(chirality_sign=-1.0)
    initial_state = IPEPS_SQUARE_C4_PT(load_square_iPEPS(args.instate_prefix, config_kwargs), config_kwargs)
    initial_state.normalize()
    template_tensor = initial_state.A_set["1,1"].copy()
    param = torch.nn.Parameter(initial_state.A_set["1,1"]._data.detach().clone())

    optimizer = LBFGS_MOD(
        [param],
        max_iter=1,
        lr=args.lr,
        tolerance_grad=1.0e-5,
        tolerance_change=1.0e-9,
        history_size=args.history_size,
        line_search_fn="backtracking",
        line_search_eps=1.0e-8,
    )

    def closure(linesearching=False):
        return compute_grad("step1_closure", optimizer, template_tensor, param, parameters, ctm_args, global_args, config_kwargs)

    @torch.no_grad()
    def closure_linesearch(linesearching=True):
        state = make_state_from_param(template_tensor, param, config_kwargs)
        loss, _ite_num, _ite_err = eval_loss(parameters, state, ctm_args, global_args, config_kwargs)
        print("line_search_E," + "{:.16g}".format(to_float(loss)))
        return loss

    optimizer.step_2c(closure, closure_linesearch)
    print("after_step1_raw_param_norm," + "{:.16g}".format(param.detach().norm().item()))

    compute_grad("step2_probe_first", optimizer, template_tensor, param, parameters, ctm_args, global_args, config_kwargs)
    compute_grad("step2_probe_second", optimizer, template_tensor, param, parameters, ctm_args, global_args, config_kwargs)


if __name__ == "__main__":
    main()
