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

Lx=6;
Ly=6;
chi=40;

global_args= GLOBALARGS()
global_args.Lx=Lx;
global_args.Ly=Ly;

ls_ctm_args= CTMARGS()
ls_ctm_args.CTM_ite_info=True
ls_ctm_args.chi=chi;
ls_ctm_args.CTM_ite_nums=10;

opt_args= OPTARGS()

init=INITCTMARGS()


# config_kwargs = {"backend": "np"}
#device: 'cpu', 'cuda'
config_kwargs = {"backend": "torch", "default_dtype": 'complex128', 'default_device': 'cuda', 'Lx':Lx, 'Ly':Ly}

filenm='SU_iPESS_Z2_csl_D4'
B_set,T_set=load_triangle_iPESS(filenm,config_kwargs);
state=IPESS_TRIANGLE(B_set,T_set,config_kwargs)
state.require_grad(False)
state.to_device('cuda')
state.normalize()
state.require_grad(True)

# print(B_set.keys())
# print(T_set.keys())
# CTM_cell=init_CTM_cell(B_set,T_set,ls_ctm_args, global_args);


# tt=CTM_cell['Cset']['1,1']['C1']
# print(tt)
# uu,C4_spec,vv=yastn.linalg.svd(tt, axes=(0, 1), svd_on_cpu=True);
# C4_spec=C4_spec.to_dense();
# print(C4_spec)

CTM0=None;
Fermionic_CTMRG_cell_iPESS(B_set,T_set,init,CTM0, ls_ctm_args, global_args);







        

