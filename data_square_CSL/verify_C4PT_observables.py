import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append("/home/sniu/python_code/iPEPS_Z2_test_codex/")
sys.path.append("D:/My Documents/Code/python_codes/iPEPS_Z2/")

import torch
import yastn

from ansatz.square_iPEPS import (
    IPEPS_SQUARE_C4_PT,
    random_square_iPEPS,
    reflect_conjugate_square_tensor_PT,
    rotate_square_tensor_C4,
)
from config.config import CTMARGS, GLOBALARGS, INITCTMARGS
from ctmrg.bosonic_CTMRG_unitcell_iPEPS import Bosonic_CTMRG_cell_iPEPS
from model.bosonic_square_ob_iPEPS import evaluate_spin_ob_cell_iPEPS


torch.set_num_threads(4)

Lx = 1
Ly = 1
D = 3
d = 2
chi = 6
device = "cpu"
checkerboard_spin_transform = "sigmay"  # options: "sigmax", "sigmay"

config_kwargs = {
    "backend": "torch",
    "default_dtype": "complex128",
    "default_device": device,
    "Lx": Lx,
    "Ly": Ly,
    "checkerboard_spin_transform": checkerboard_spin_transform,
}

global_args = GLOBALARGS()
global_args.Lx = Lx
global_args.Ly = Ly
global_args.device = device

ctm_args = CTMARGS()
ctm_args.CTM_ite_info = True
ctm_args.chi = chi
ctm_args.CTM_ite_nums = 20
ctm_args.CTM_trun_tol = 1e-8
ctm_args.doublelayer_on_cpu = False
ctm_args.use_sub_checkpoint = False

A_set = random_square_iPEPS(D, d, config_kwargs)
state = IPEPS_SQUARE_C4_PT(A_set, config_kwargs, impose_c4=True, impose_pt=True)
state.require_grad(False)
state.to_device(device)
state.normalize()

A = state.A_set["1,1"]
c4_residual = yastn.linalg.norm(A - rotate_square_tensor_C4(A, 1)).item()
pt_residual = yastn.linalg.norm(A - reflect_conjugate_square_tensor_PT(A)).item()
print("tensor C4 residual:", c4_residual)
print("tensor PT residual:", pt_residual)

CTM_cell, double_A_cell, ite_num, ite_err = Bosonic_CTMRG_cell_iPEPS(state.A_set, INITCTMARGS(), None, ctm_args, global_args)
print("CTM ite_num:", ite_num)
print("CTM ite_err:", ite_err)

(
    triangle_no_LU_set,
    triangle_no_LD_set,
    triangle_no_RD_set,
    triangle_no_RU_set,
    SS_x_set,
    SS_y_set,
    SS_diag_a_set,
    SS_diag_b_set,
) = evaluate_spin_ob_cell_iPEPS(state.A_set, double_A_cell, CTM_cell, config_kwargs, global_args)

SS_x = SS_x_set[0, 0]
SS_y = SS_y_set[0, 0]
SS_diag_a = SS_diag_a_set[0, 0]
SS_diag_b = SS_diag_b_set[0, 0]
triangles = torch.stack(
    [
        triangle_no_LU_set[0, 0],
        triangle_no_LD_set[0, 0],
        triangle_no_RD_set[0, 0],
        triangle_no_RU_set[0, 0],
    ]
)

print("SS_x:", SS_x)
print("SS_y:", SS_y)
print("SS_diag_a:", SS_diag_a)
print("SS_diag_b:", SS_diag_b)
print("chirality no_LU/no_LD/no_RD/no_RU:", triangles)

ss_xy_diff = torch.abs(SS_x - SS_y).item()
diag_diff = torch.abs(SS_diag_a - SS_diag_b).item()
triangle_diff = torch.max(torch.abs(triangles - torch.mean(triangles))).item()

print("C4 diff |SS_x-SS_y|:", ss_xy_diff)
print("C4 diff |SS_diag_a-SS_diag_b|:", diag_diff)
print("C4 diff max triangle deviation:", triangle_diff)

tensor_tol = 1.0e-10
observable_tol = 1.0e-5
if c4_residual > tensor_tol:
    raise RuntimeError("tensor C4 residual is too large")
if pt_residual > tensor_tol:
    raise RuntimeError("tensor PT residual is too large")
if ss_xy_diff > observable_tol:
    raise RuntimeError("nearest-neighbor observables do not satisfy C4")
if diag_diff > observable_tol:
    raise RuntimeError("diagonal observables do not satisfy C4")
if triangle_diff > observable_tol:
    raise RuntimeError("chirality observables do not satisfy C4")
