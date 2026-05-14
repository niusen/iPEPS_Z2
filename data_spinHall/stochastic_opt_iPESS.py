import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Must be set BEFORE importing torch
print("PYTORCH_CUDA_ALLOC_CONF:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
import sys
sys.path.append('/home/sniu/python_code/iPEPS_Z2_test_codex/')
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
from optimization.stochastic_opt import *

########################
pid = os.getpid();
print('pid= '+str(pid))
n_cpu=20;
torch.set_num_threads(n_cpu)
########################
t1=1;
t2=1;
ϕ=numpy.pi/2;
μ=0;
U=0;
mx=0;
parameters={"t1": t1, "t2": t2, "ϕ": ϕ, "μ":  μ, "U":  U, "mx":  mx};
print(parameters)
energy_setting=Square_Hubbard_Energy_settings();
energy_setting.model = 'triangle_spinHall';

Noise=0;
print(Noise);

Lx=2;
Ly=2;
D=4;
chi=40;



AD_ctm_args= CTMARGS()
AD_ctm_args.CTM_ite_info=True
AD_ctm_args.chi=chi;
AD_ctm_args.CTM_ite_nums=10;
AD_ctm_args.CTM_trun_tol=1e-8
AD_ctm_args.doublelayer_on_cpu=True;
AD_ctm_args.use_sub_checkpoint=True;
print(AD_ctm_args)

ls_ctm_args= CTMARGS()
ls_ctm_args.CTM_ite_info=False
ls_ctm_args.chi=chi;
ls_ctm_args.CTM_ite_nums=50;
ls_ctm_args.CTM_trun_tol=1e-8
print(ls_ctm_args)

opt_args= OPTARGS()

init=INITCTMARGS()


# config_kwargs = {"backend": "np"}
#device: 'cpu', 'cuda'
config_kwargs = {"backend": 'torch', "default_dtype": 'complex128', 'default_device': 'cuda:1', 'Lx':Lx, 'Ly':Ly}

global_args= GLOBALARGS()
global_args.Lx=Lx;
global_args.Ly=Ly;
global_args.device=config_kwargs['default_device'];

#filenm='SU_iPESS_Z2_csl_D'+str(D);
filenm='Z2_D4_chi40_-1.616'
B_set,T_set=load_triangle_iPESS(filenm,config_kwargs);
state=IPESS_TRIANGLE(B_set,T_set,config_kwargs)

state=add_noise(state,Noise,config_kwargs);

state.require_grad(False)
state.to_device(config_kwargs['default_device'])
state.normalize()
# state.require_grad(True)

# print(B_set.keys())
# print(T_set.keys())
B_set=state.B_set
T_set=state.T_set
print(B_set['1,1'].requires_grad)

# a,b=yastn.Tensor.compress_to_1d(B_set['1,1'])
# tnew=yastn.decompress_from_1d(a,b)

# CTM_cell=init_CTM_cell(B_set,T_set,ls_ctm_args, global_args);


# Ident, N_occu, n_double, Cdag, C, CdagC_string=Hamiltonians_spinful_Z2(config_kwargs)
#sx,sy,sz=spin_operator_Z2(config_kwargs);
#Sa, Sb, SS_string, chirality_S1, chirality_S2, chirality_S3, chirality_string12, chirality_string23=Operators_spinful_Z2(config_kwargs);


# print()




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



opt_method='lbfgs'; # options: 'stochastic', 'cg', 'lbfgs'

ls=LINESEARCH()
ls.method=opt_method;
ls.maxiter=100;
ls.gtol=1e-5;

if opt_method=='stochastic':
    ls.delta0=1e-3;
    ls.alpha=3/4;
    stochastic_opt(parameters,D,chi, state, AD_ctm_args, ls_ctm_args, energy_setting, global_args, config_kwargs,  ls)
elif opt_method=='cg':
    ls.cg_beta='PRP'; # options: 'PRP', 'FR'
    ls.line_search='hager_zhang'; # options: 'hager_zhang', 'backtracking'
    optimize_iPESS(parameters,D,chi, state, AD_ctm_args, ls_ctm_args, energy_setting, global_args, config_kwargs,  ls)
elif opt_method=='lbfgs':
    ls.line_search='hager_zhang'; # options: 'hager_zhang', 'backtracking'
    optimize_iPESS(parameters,D,chi, state, AD_ctm_args, ls_ctm_args, energy_setting, global_args, config_kwargs,  ls)
else:
    raise ValueError("unknown optimization method: "+str(opt_method))







        

