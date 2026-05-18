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
from optimization.optimize_bosonic_square_iPEPS import _double_layer_for_observables


def to_float(x):
    if hasattr(x, "item"):
        x = x.item()
    return float(numpy.real(x))


def eval_loss(parameters, state, ctm_args, global_args, config_kwargs):
    CTM_cell, double_A_cell, ite_num, ite_err = Bosonic_CTMRG_cell_iPEPS(
        state.A_set, INITCTMARGS(), None, ctm_args, global_args
    )
    double_A_cell = _double_layer_for_observables(double_A_cell, ctm_args, global_args)
    loss, *_ = evaluate_ob_cell_iPEPS(parameters, state.A_set, double_A_cell, CTM_cell, config_kwargs, global_args)
    return torch.real(loss), ite_num, ite_err


def dense_array(tensor):
    return tensor.to_dense().detach().cpu().numpy().copy()


def norm_array(arr):
    return float(numpy.linalg.norm(arr.reshape(-1)))


def main():
    print("pid= " + str(os.getpid()))
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-prefix", required=True)
    parser.add_argument("--raw-npz", required=True)
    parser.add_argument("--projected-prefix", default=None)
    parser.add_argument("--chi", type=int, default=40)
    parser.add_argument("--ctm-max-iter", type=int, default=80)
    parser.add_argument("--ctm-conv-tol", type=float, default=1.0e-8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--optimizer-context", action="store_true")
    parser.add_argument("--ctm-trun-tol", type=float, default=1.0e-8)
    parser.add_argument("--svd-ad-decomp-reg", type=float, default=None)
    parser.add_argument("--projector-min-singular-cutoff", type=float, default=None)
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

    ctm_args = CTMARGS()
    ctm_args.chi = args.chi
    ctm_args.CTM_ite_nums = args.ctm_max_iter
    ctm_args.CTM_trun_tol = args.ctm_trun_tol
    ctm_args.CTM_conv_tol = args.ctm_conv_tol
    ctm_args.CTM_ite_info = False
    ctm_args.doublelayer_on_cpu = False
    ctm_args.use_sub_checkpoint = False
    if args.svd_ad_decomp_reg is not None:
        ctm_args.svd_ad_decomp_reg = args.svd_ad_decomp_reg
    if args.projector_min_singular_cutoff is not None:
        ctm_args.projector_min_singular_cutoff = args.projector_min_singular_cutoff

    parameters = prl_129_177201_square_csl_parameters(chirality_sign=-1.0)
    template_state = IPEPS_SQUARE_C4_PT(
        load_square_iPEPS(args.template_prefix, config_kwargs),
        config_kwargs,
        impose_c4=True,
        impose_pt=True,
    )
    template_state.to_device(args.device)
    template_state.normalize()
    template = template_state.A_set["1,1"].copy()

    raw_data = numpy.load(args.raw_npz)
    raw_param_saved = raw_data["raw_param"]
    saved_grad = raw_data["raw_grad"]
    raw_param = torch.nn.Parameter(torch.as_tensor(raw_param_saved, dtype=template._data.dtype, device=args.device))

    raw_tensor = template._replace(data=raw_param)
    projected_no_norm = IPEPS_SQUARE_C4_PT(
        OrderedDict({"1,1": raw_tensor}),
        config_kwargs,
        impose_c4=True,
        impose_pt=True,
    )
    projected_dense_before_norm = dense_array(projected_no_norm.A_set["1,1"])
    projected_no_norm_norm = norm_array(projected_dense_before_norm)

    raw_state = projected_no_norm
    raw_state.normalize()
    if args.optimizer_context:
        with torch.no_grad():
            with torch.enable_grad():
                raw_loss, raw_steps, raw_err = eval_loss(parameters, raw_state, ctm_args, global_args, config_kwargs)
                raw_loss.backward()
    else:
        raw_loss, raw_steps, raw_err = eval_loss(parameters, raw_state, ctm_args, global_args, config_kwargs)
        raw_loss.backward()
    raw_grad = raw_param.grad.detach().cpu().numpy().copy()

    clean_tensor = raw_state.A_set["1,1"].copy()
    clean_param = torch.nn.Parameter(clean_tensor._data.detach().clone())
    clean_state = IPEPS_SQUARE_C4_PT(
        OrderedDict({"1,1": clean_tensor._replace(data=clean_param)}),
        config_kwargs,
        impose_c4=False,
        impose_pt=False,
    )
    clean_loss, clean_steps, clean_err = eval_loss(parameters, clean_state, ctm_args, global_args, config_kwargs)
    clean_loss.backward()
    clean_grad = clean_param.grad.detach().cpu().numpy().copy()

    print("raw_npz,", args.raw_npz)
    print("optimizer_context,", args.optimizer_context)
    print("ctm_trun_tol,", f"{ctm_args.CTM_trun_tol:.16g}")
    print("svd_ad_decomp_reg,", f"{ctm_args.svd_ad_decomp_reg:.16g}")
    print("projector_min_singular_cutoff,", f"{ctm_args.projector_min_singular_cutoff:.16g}")
    print("saved_raw_loss,", f"{float(raw_data['loss']):.16g}")
    print("saved_raw_grad_norm,", f"{float(raw_data['raw_grad_norm']):.16g}")
    print("recomputed_raw_loss,", f"{to_float(raw_loss):.16g}")
    print("clean_loss,", f"{to_float(clean_loss):.16g}")
    print("raw_ctm_steps,", raw_steps)
    print("raw_ctm_err,", raw_err)
    print("clean_ctm_steps,", clean_steps)
    print("clean_ctm_err,", clean_err)
    print("raw_param_norm_flat,", f"{norm_array(raw_param_saved):.16g}")
    print("projected_no_norm_norm,", f"{projected_no_norm_norm:.16g}")
    print("projected_normalized_norm,", f"{norm_array(dense_array(raw_state.A_set['1,1'])):.16g}")
    print("raw_grad_norm_recomputed,", f"{norm_array(raw_grad):.16g}")
    print("raw_grad_saved_recomputed_diff,", f"{norm_array(raw_grad - saved_grad):.16g}")
    print("clean_grad_norm,", f"{norm_array(clean_grad):.16g}")
    print("raw_over_clean_grad_norm,", f"{norm_array(raw_grad) / max(norm_array(clean_grad), 1.0e-300):.16g}")
    if args.projected_prefix is not None:
        projected_state = IPEPS_SQUARE_C4_PT(
            load_square_iPEPS(args.projected_prefix, config_kwargs),
            config_kwargs,
            impose_c4=True,
            impose_pt=True,
        )
        projected_state.normalize()
        projected_diff = dense_array(projected_state.A_set["1,1"]) - dense_array(raw_state.A_set["1,1"])
        print("projected_json_diff_norm,", f"{norm_array(projected_diff):.16g}")
        print("projected_json_diff_max,", f"{float(numpy.max(numpy.abs(projected_diff))):.16g}")


if __name__ == "__main__":
    main()
