import os
import sys
sys.path.append('D:/My Documents/Code/python_codes/iPEPS_Z2')
from collections import OrderedDict
import json
import numpy
import torch
import yastn
from config.settings import *
from config.config import *
from ansatz.triangle_iPESS import *
from ctmrg.Fermionic_CTMRG_unitcell_iPESS import *
from model.fermion_ob_iPESS import *
from model.iPESS_correl_cell import *
from scipy.io import savemat

########################
pid = os.getpid();
print('pid= '+str(pid))
n_cpu=10;
torch.set_num_threads(n_cpu)
########################
t1=1;
t2=1;
ϕ=numpy.pi/2;
μ=0;
U=20;
B=0;
parameters={"t1": t1, "t2": t2, "ϕ": ϕ, "μ":  μ, "U":  U, "B":  B};
print(parameters)
energy_setting=Square_Hubbard_Energy_settings();
energy_setting.model = 'spinful_triangle_lattice';

Lx=6;
Ly=6;
D=4;
chi=40;

global_args= GLOBALARGS()
global_args.Lx=Lx;
global_args.Ly=Ly;



ls_ctm_args= CTMARGS()
ls_ctm_args.CTM_ite_info=True
ls_ctm_args.chi=chi;
ls_ctm_args.CTM_ite_nums=30;
ls_ctm_args.CTM_trun_tol=1e-8
print(ls_ctm_args)

opt_args= OPTARGS()

init=INITCTMARGS()


# config_kwargs = {"backend": "np"}
#device: 'cpu', 'cuda'
config_kwargs = {"backend": 'torch', "default_dtype": 'complex128', 'default_device': 'cuda', 'Lx':Lx, 'Ly':Ly}
global_args.device=config_kwargs['default_device'];

filenm='SU_iPESS_Z2_csl_D'+str(D);
B_set,T_set=load_triangle_iPESS(filenm,config_kwargs);
state=IPESS_TRIANGLE(B_set,T_set,config_kwargs)
state.require_grad(False)
state.to_device(global_args.device)
state.normalize()



with torch.no_grad():

    B_set=state.B_set
    T_set=state.T_set

    chis=[40,80,120,160]

    init=INITCTMARGS()
    CTM_cell=None

    for chi in chis:
        ls_ctm_args.chi=chi;
        CTM_cell, double_B_set,double_T_set,ite_num,ite_err=Fermionic_CTMRG_cell_iPESS(B_set,T_set,init,CTM_cell, ls_ctm_args, global_args);

        E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell_iPESS(parameters, B_set,T_set, double_B_set, double_T_set, CTM_cell, energy_setting, config_kwargs, global_args);
        print('E= '+str(E_total.item()))
        print(ex_set.tolist())
        print(ey_set.tolist())
        print(e_diagonala_set.tolist())
        print(e0_set.tolist())
        print(eU_set.tolist())

        pairing_x_set,pairing_y_set,pairing_diagonal_set=evaluate_ob_pairing_cell(B_set,T_set, double_B_set, double_T_set, CTM_cell, config_kwargs, global_args);
        print(pairing_x_set)
        print(pairing_y_set)
        print(pairing_diagonal_set)

        sx_set,sy_set,sz_set=evaluate_spin_cell_iPESS(B_set,T_set, double_B_set, double_T_set, CTM_cell, config_kwargs, global_args);
        print(sx_set)
        print(sy_set)
        print(sz_set)


        triangle_up_set,triangle_dn_set,SS_x_set,SS_y_set,SS_diagonal_set=evaluate_spin_ob_cell_iPESS(B_set,T_set, double_B_set, double_T_set, CTM_cell, energy_setting, config_kwargs, global_args);
        print(triangle_up_set)
        print(triangle_dn_set)
        print(SS_x_set)
        print(SS_y_set)
        print(SS_diagonal_set)

        mat_filenm="ob_D"+str(D)+"_chi"+str(chi);
        datadic = {"E_total": E_total.item(), "e0_set": e0_set.cpu().numpy(), "eU_set": eU_set.cpu().numpy(), "ex_set": ex_set.cpu().numpy(), "ey_set": ey_set.cpu().numpy(), "e_diagonala_set":e_diagonala_set.cpu().numpy(), "pairing_x_set":pairing_x_set.cpu().numpy(), "pairing_y_set":pairing_y_set.cpu().numpy(), "pairing_diagonal_set":pairing_diagonal_set.cpu().numpy(),  "sx_set": sx_set.cpu().numpy(), "sy_set": sy_set.cpu().numpy(), "sz_set":sz_set.cpu().numpy(), "triangle_up_set":triangle_up_set.cpu().numpy(), "triangle_dn_set":triangle_dn_set.cpu().numpy(), "SS_x_set":SS_x_set.cpu().numpy(), "SS_y_set":SS_y_set.cpu().numpy(),"SS_diagonal_set":SS_diagonal_set.cpu().numpy() }
        savemat(mat_filenm+".mat", datadic)


        distance=40;
        partly=True;
        SS_ob_set,CdagC_ob_set=cal_correl(CTM_cell,B_set,T_set,double_B_set, double_T_set,D,chi,'x',distance, config_kwargs, global_args, partly);

        init.reconstruct_CTM=False




        
