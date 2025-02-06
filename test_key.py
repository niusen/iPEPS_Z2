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

n_cpu=10;
torch.set_num_threads(n_cpu)

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
print(ls_ctm_args, flush=True)

opt_args= OPTARGS()

init=INITCTMARGS()


# config_kwargs = {"backend": "np"}
#device: 'cpu', 'cuda'
config_kwargs = {"backend": 'torch', "default_dtype": 'complex128', 'default_device': 'cuda', 'Lx':Lx, 'Ly':Ly}

filenm='Z2_D4_chi40'
with open(filenm+'.json') as f:
    data = json.load(f)
    # print(data)
B_set=data['B_set'];
T_set=data['T_set'];
B_set_new=OrderedDict()
T_set_new=OrderedDict()
for cx in range(1,Lx+1):
    for cy in range(1,Ly+1):
        B_set_new.update({str(cx)+','+str(cy): B_set[str(cx+1)+','+str(cy+1)]})
        T_set_new.update({str(cx)+','+str(cy): T_set[str(cx+1)+','+str(cy+1)]})

with open(filenm+'new.json', "w") as f:
    json.dump({'T_set':T_set_new,'B_set':B_set_new}, f)