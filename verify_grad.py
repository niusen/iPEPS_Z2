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

#############################
#memory_limit = 15;#GB
#print('restricted memory:'+str(memory_limit)+'GB')
#resource.setrlimit(resource.RLIMIT_AS, (memory_limit*1024 * 1024 * 1024, memory_limit*1024 * 1024 * 1024))
n_cpu=10;
torch.set_num_threads(n_cpu)
#############################

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
D=4;
chi=40;



ls_ctm_args= CTMARGS()
ls_ctm_args.CTM_ite_info=True
ls_ctm_args.chi=chi;
ls_ctm_args.CTM_ite_nums=1;
ls_ctm_args.CTM_trun_tol=1e-8
ls_ctm_args.use_checkpoint=True;
ls_ctm_args.checkpoint_device='cpu';
print(ls_ctm_args, flush=True)

opt_args= OPTARGS()

init=INITCTMARGS()


# config_kwargs = {"backend": "np"}
#device: 'cpu', 'cuda'
config_kwargs = {"backend": 'torch', "default_dtype": 'complex128', 'default_device': 'cuda', 'Lx':Lx, 'Ly':Ly}

global_args= GLOBALARGS()
global_args.Lx=Lx;
global_args.Ly=Ly;
global_args.device=config_kwargs['default_device'];

filenm='Z2_D4_chi40'
B_set,T_set=load_triangle_iPESS(filenm,config_kwargs);
state=IPESS_TRIANGLE(B_set,T_set,config_kwargs)
state.requires_grad_(False)
state.to_device(config_kwargs['default_device'])
state.normalize()
# state.require_grad(True)

# print(B_set.keys())
# print(T_set.keys())
# B_set=state.B_set
# T_set=state.T_set
print(state.requires_grad)

# a,b=yastn.Tensor.compress_to_1d(B_set['1,1'])
# tnew=yastn.decompress_from_1d(a,b)

# CTM_cell=init_CTM_cell(B_set,T_set,ls_ctm_args, global_args);


# Ident, N_occu, n_double, Cdag, C, CdagC_string=Hamiltonians_spinful_Z2(config_kwargs)
#sx,sy,sz=spin_operator_Z2(config_kwargs);
#Sa, Sb, SS_string, chirality_S1, chirality_S2, chirality_S3, chirality_string12, chirality_string23=Operators_spinful_Z2(config_kwargs);


# print()
def cost_fun(state, ls_ctm_args, energy_setting, global_args, config_kwargs):
    state.requires_grad_(True)
    # B_set=state.B_set
    # T_set=state.T_set

    CTM0=None;
    # CTM_cell, double_B_set,double_T_set,ite_num,ite_err=Fermionic_CTMRG_cell_iPESS(state,init,CTM0, ls_ctm_args, global_args);
    CTM_cell, state_double_layer,ite_num,ite_err=Fermionic_CTMRG_cell_iPESS(state,init,CTM0, ls_ctm_args, global_args);

    E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell_iPESS(parameters, state, state_double_layer, CTM_cell, energy_setting, config_kwargs, global_args);
    # print(E_total)
    # print(ex_set)
    # print(ey_set)
    # print(e_diagonala_set)
    # print(e0_set)
    # print(eU_set)
    return E_total
def get_grad(state, ls_ctm_args, energy_setting, global_args, config_kwargs):
    E=cost_fun(state, ls_ctm_args, energy_setting, global_args, config_kwargs)
    E.backward()
    coord='1,1';
    grad_B=state.B_set[coord].grad()
    a,b=yastn.Tensor.compress_to_1d(grad_B)
    print(a)
    grad_T=state.T_set[coord].grad()
    a,b=yastn.Tensor.compress_to_1d(grad_T)
    print(a)

get_grad(state, ls_ctm_args, energy_setting, global_args, config_kwargs)

def finite_diff(state, ls_ctm_args, energy_setting, global_args, config_kwargs):
    print('finite diff:',  flush=True)
    E0=cost_fun(state, ls_ctm_args, energy_setting, global_args, config_kwargs)
    coord='1,1'
    ls_ctm_args.CTM_ite_info=False
    
    a,b=yastn.Tensor.compress_to_1d(state.B_set[coord]);
    grad_=copy.deepcopy(a)*0;
    L=len(a);
    delta=1e-6;
    for cc in range(0,L):
        state1=state.copy();
        bm=state1.B_set[coord];
        data_1d,b=yastn.Tensor.compress_to_1d(bm);
        data_1d[cc]=data_1d[cc]+delta
        bmnew=yastn.decompress_from_1d(data_1d,b)
        state1.B_set[coord]=bmnew;
        E=cost_fun(state1, ls_ctm_args, energy_setting, global_args, config_kwargs)
        grad_[cc]=grad_[cc]+(E-E0)/delta

        state1=state.copy();
        bm=state1.B_set[coord];
        data_1d,b=yastn.Tensor.compress_to_1d(bm);
        data_1d[cc]=data_1d[cc]+delta*1j
        bmnew=yastn.decompress_from_1d(data_1d,b)
        state1.B_set[coord]=bmnew;
        E=cost_fun(state1, ls_ctm_args, energy_setting, global_args, config_kwargs)
        grad_[cc]=grad_[cc]+(E-E0)/delta*1j
        
        print(grad_[cc].item())
    print(grad_)
    
# finite_diff(state, ls_ctm_args, energy_setting, global_args, config_kwargs)

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

