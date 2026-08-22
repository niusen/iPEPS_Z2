import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(r"D:\My Documents\Code\python_codes\iPEPS_Z2")

import torch

from config.config import CTMARGS, GLOBALARGS
from data_square_CSL.diagnose_C4PT_gradient import make_state_from_juraj
from model.bosonic_square_ob_iPEPS import prl_129_177201_square_csl_parameters
from optimization.optimize_bosonic_square_iPEPS import get_grad


print("pid= " + str(os.getpid()))
torch.set_num_threads(4)

config_kwargs = {
    "backend": "torch",
    "default_dtype": "complex128",
    "default_device": "cpu",
    "Lx": 1,
    "Ly": 1,
    "checkerboard_spin_transform": "sigmay",
}

global_args = GLOBALARGS()
global_args.Lx = 1
global_args.Ly = 1
global_args.device = "cpu"

ctm_args = CTMARGS()
ctm_args.chi = 40
ctm_args.CTM_ite_nums = 10
ctm_args.CTM_trun_tol = 1.0e-8
ctm_args.CTM_conv_tol = 0.0
ctm_args.CTM_ite_info = False
ctm_args.doublelayer_on_cpu = False
ctm_args.use_sub_checkpoint = False

parameters = prl_129_177201_square_csl_parameters(chirality_sign=-1.0)
state = make_state_from_juraj(
    r"D:\My Documents\Code\python_codes\Juraj\tn-torch_dev\test_csl\prl129_D3_chi40_state.json",
    config_kwargs,
)

for step in range(1, 4):
    grad, energy, _, _ = get_grad(parameters, state, ctm_args, None, global_args, config_kwargs, return_ctm_err=True)
    print(f"repeat={step}, energy={energy.item():.16g}, grad_norm={grad.norm():.16g}")
