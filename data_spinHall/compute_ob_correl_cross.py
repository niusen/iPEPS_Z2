import os
import sys
from pathlib import Path

def _find_repo_root(start):
    for path in [start] + list(start.parents):
        if (path / "ansatz").is_dir() and (path / "model").is_dir() and (path / "config").is_dir():
            return path
    raise RuntimeError("Could not locate iPEPS_Z2 root directory from " + str(start))

sys.path.insert(0, str(_find_repo_root(Path(__file__).resolve().parent)))

import numpy
import torch
from scipy.io import savemat

from ansatz.triangle_iPESS import IPESS_TRIANGLE, load_triangle_iPESS
from config.config import CTMARGS, GLOBALARGS, INITCTMARGS, OPTARGS, Square_Hubbard_Energy_settings
from ctmrg.Fermionic_CTMRG_unitcell_iPESS import Fermionic_CTMRG_cell_iPESS
from model.fermion_ob_iPESS import evaluate_ob_cell_iPESS, evaluate_spin_ob_cell_iPESS
from model.iPESS_correl_cell_cross_copy import (
    cal_correl_density_cross_copy,
    cal_correl_spin_resolved_cross_copy,
    cal_onsite_cross_copy,
)


pid = os.getpid()
print("pid= " + str(pid))
n_cpu = 10
torch.set_num_threads(n_cpu)

t1=1;
t2=1;
ϕ=numpy.pi/2;
μ=0;
U=0;
B=0;
mx=0;
parameters={"t1": t1, "t2": t2, "ϕ": ϕ, "μ":  μ, "U":  U, "B":  B, "mx":  mx};
print(parameters)
energy_setting = Square_Hubbard_Energy_settings()
energy_setting.model = "triangle_spinHall"

Lx = 2
Ly = 1
D = 4
chi = 40

global_args = GLOBALARGS()
global_args.Lx = Lx
global_args.Ly = Ly

ls_ctm_args = CTMARGS()
ls_ctm_args.CTM_ite_info = True
ls_ctm_args.chi = chi
ls_ctm_args.CTM_ite_nums = 30
ls_ctm_args.CTM_trun_tol = 1.0e-10
print(ls_ctm_args)

opt_args = OPTARGS()
init = INITCTMARGS()

config_kwargs = {"backend": "torch", "default_dtype": "complex128", "default_device": "cpu", "Lx": Lx, "Ly": Ly}
global_args.device = config_kwargs["default_device"]

filenm = "Z2_D4_chi40_-2.38533"
B_set, T_set = load_triangle_iPESS(filenm, config_kwargs)
state = IPESS_TRIANGLE(B_set, T_set, config_kwargs)
state.require_grad(False)
state.to_device(config_kwargs["default_device"])
state.normalize()

with torch.no_grad():
    B_set = state.B_set
    T_set = state.T_set

    chis = [40, 80, 120, 160]
    init = INITCTMARGS()
    CTM_cell = None

    for chi in chis:
        ls_ctm_args.chi = chi
        CTM_cell, double_B_set, double_T_set, ite_num, ite_err = Fermionic_CTMRG_cell_iPESS(
            B_set, T_set, init, CTM_cell, ls_ctm_args, global_args
        )

        (
            E_total,
            ex_up_set,
            ey_up_set,
            e_diagonala_up_set,
            ex_dn_set,
            ey_dn_set,
            e_diagonala_dn_set,
            e0_set,
            eU_set,
            sx_set,
            sy_set,
            sz_set,
        ) = evaluate_ob_cell_iPESS(
            parameters, B_set, T_set, double_B_set, double_T_set, CTM_cell, energy_setting, config_kwargs, global_args
        )
        print("E= " + str(E_total.item()))
        print(ex_up_set.tolist())
        print(ey_up_set.tolist())
        print(e_diagonala_up_set.tolist())
        print(ex_dn_set.tolist())
        print(ey_dn_set.tolist())
        print(e_diagonala_dn_set.tolist())
        print(e0_set.tolist())
        print(eU_set.tolist())
        print(sx_set)
        print(sy_set)
        print(sz_set)

        triangle_up_set, triangle_dn_set, SS_x_set, SS_y_set, SS_diagonal_set = evaluate_spin_ob_cell_iPESS(
            B_set, T_set, double_B_set, double_T_set, CTM_cell, energy_setting, config_kwargs, global_args
        )
        print(triangle_up_set)
        print(triangle_dn_set)
        print(SS_x_set)
        print(SS_y_set)
        print(SS_diagonal_set)

        mat_filenm = "ob_spinHall_cross_base_D" + str(D) + "_chi" + str(chi)
        datadic = {
            "E_total": E_total.item(),
            "e0_set": e0_set.cpu().numpy(),
            "eU_set": eU_set.cpu().numpy(),
            "ex_up_set": ex_up_set.cpu().numpy(),
            "ey_up_set": ey_up_set.cpu().numpy(),
            "e_diagonala_up_set": e_diagonala_up_set.cpu().numpy(),
            "ex_dn_set": ex_dn_set.cpu().numpy(),
            "ey_dn_set": ey_dn_set.cpu().numpy(),
            "e_diagonala_dn_set": e_diagonala_dn_set.cpu().numpy(),
            "sx_set": sx_set.cpu().numpy(),
            "sy_set": sy_set.cpu().numpy(),
            "sz_set": sz_set.cpu().numpy(),
            "SS_x_set": SS_x_set.cpu().numpy(),
            "SS_y_set": SS_y_set.cpu().numpy(),
            "SS_diagonal_set": SS_diagonal_set.cpu().numpy(),
        }
        savemat(mat_filenm + ".mat", datadic)

        distance = 40
        partly = True
        cal_onsite_cross_copy(
            CTM_cell,
            B_set,
            T_set,
            double_B_set,
            double_T_set,
            D,
            chi,
            config_kwargs,
            global_args,
        )
        cal_correl_spin_resolved_cross_copy(
            CTM_cell,
            B_set,
            T_set,
            double_B_set,
            double_T_set,
            D,
            chi,
            "x",
            distance,
            config_kwargs,
            global_args,
            partly,
        )
        cal_correl_density_cross_copy(
            CTM_cell,
            B_set,
            T_set,
            double_B_set,
            double_T_set,
            D,
            chi,
            "x",
            distance,
            config_kwargs,
            global_args,
            partly,
        )

        init.reconstruct_CTM = False
