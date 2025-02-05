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

t1=1;
t2=1;
ϕ=numpy.pi/2;
μ=0;
U=20;
B=0;
parameters={"t1": t1, "t2": t2, "ϕ": ϕ, "μ":  μ, "U":  U, "B":  B};

energy_setting=Square_Hubbard_Energy_settings();
energy_setting.model = 'spinful_triangle_lattice';

Lx=6;
Ly=6;
chi=40;

global_args= GLOBALARGS()
global_args.Lx=Lx;
global_args.Ly=Ly;

ls_ctm_args= CTMARGS()
ls_ctm_args.CTM_ite_info=True
ls_ctm_args.chi=chi;
ls_ctm_args.CTM_ite_nums=1;

opt_args= OPTARGS()

init=INITCTMARGS()


# config_kwargs = {"backend": "np"}
#device: 'cpu', 'cuda'
config_kwargs = {"backend": 'torch', "default_dtype": 'complex128', 'default_device': 'cuda', 'Lx':Lx, 'Ly':Ly}

filenm='SU_iPESS_Z2_csl_D4'
B_set,T_set=load_triangle_iPESS(filenm,config_kwargs);
state=IPESS_TRIANGLE(B_set,T_set,config_kwargs)
state.require_grad(False)
state.to_device('cuda')
state.normalize()
state.require_grad(True)

# print(B_set.keys())
# print(T_set.keys())
B_set=state.B_set
T_set=state.T_set
print(B_set['1,1'].requires_grad)

# a,b=yastn.Tensor.compress_to_1d(B_set['1,1'])
# tnew=yastn.decompress_from_1d(a,b)

CTM_cell=init_CTM_cell(B_set,T_set,ls_ctm_args, global_args);


# Ident, N_occu, n_double, Cdag, C, CdagC_string=Hamiltonians_spinful_Z2(config_kwargs)
#sx,sy,sz=spin_operator_Z2(config_kwargs);
#Sa, Sb, SS_string, chirality_S1, chirality_S2, chirality_S3, chirality_string12, chirality_string23=Operators_spinful_Z2(config_kwargs);


# print()

CTM0=None;
CTM_cell, double_B_set,double_T_set,ite_num,ite_err=Fermionic_CTMRG_cell_iPESS(B_set,T_set,init,CTM_cell, ls_ctm_args, global_args);

E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell_iPESS(parameters, B_set,T_set, double_B_set, double_T_set, CTM_cell, energy_setting, config_kwargs, global_args);
print(E_total)
print(ex_set)
print(ey_set)
print(e_diagonala_set)
print(e0_set)
print(eU_set)

E_total.backward()
print(B_set['1,1'].grad().to_dense())
# sx_set,sy_set,sz_set=evaluate_spin_cell_iPESS(B_set,T_set, double_B_set, double_T_set, CTM_cell, config_kwargs, global_args);
# print(sx_set)
# print(sy_set)
# print(sz_set)


# triangle_up_set,triangle_dn_set,SS_x_set,SS_y_set,SS_diagonal_set=evaluate_spin_ob_cell_iPESS(B_set,T_set, double_B_set, double_T_set, CTM_cell, energy_setting, config_kwargs, global_args);
# print(triangle_up_set)
# print(triangle_dn_set)
# print(SS_x_set)
# print(SS_y_set)
# print(SS_diagonal_set)




        

