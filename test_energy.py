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
CTM_cell=init_CTM_cell(B_set,T_set,ls_ctm_args, global_args);

def spin_operator_Z2(config_kwargs):
    
    config_Z2 = yastn.make_config(sym='Z2',fermionic=True, **config_kwargs)
    Device=config_kwargs['default_device'];

    Vp = yastn.Leg(config_Z2, s=1, t=(0, 1), D=(2, 2))
    Vp_conj = yastn.Leg(config_Z2, s=-1, t=(0, 1), D=(2, 2))


    Id=torch.tensor([[1.0, 0], [0, 1.0]]).to(device=Device);
    sm=torch.tensor([[0, 1.0], [0, 0]]).to(device=Device); 
    sp=torch.tensor([[0, 0], [1.0, 0]]).to(device=Device);
    sz=torch.tensor([[1.0, 0], [0, -1.0]]).to(device=Device); 
    occu=torch.tensor([[0, 0], [0, 1.0]]).to(device=Device);
    
    #order of kron() command: (0,0), (0,1), (1,0), (1,1)
    order=(1-1,4-1,3-1,2-1);
    
    Cdagup_Cup=torch.zeros((4,4),dtype=Id.dtype,device=Id.device);
    Cdagup_Cup=torch.kron(sp*sm,Id);
    Cdagup_Cup=Cdagup_Cup[order,:]
    Cdagup_Cup=Cdagup_Cup[:,order]
    Cdagup_Cup_=yastn.zeros(config=config_Z2, legs=[Vp,Vp_conj])
    Cdagup_Cup_=fill_Z2_Ham(Cdagup_Cup,Cdagup_Cup_)

    Cdagdn_Cdn=zeros(4,4);
    Cdagdn_Cdn[[1,4,3,2],[1,4,3,2]]=kron(Id,sp*sm);
    Cdagdn_Cdn=TensorMap(Cdagdn_Cdn,  V ← V);

    Cdagup_Cdn=zeros(4,4);
    Cdagup_Cdn[[1,4,3,2],[1,4,3,2]]=kron(sp,sm);
    Cdagup_Cdn=TensorMap(Cdagup_Cdn,  V ← V);

    Cdagdn_Cup=zeros(4,4);
    Cdagdn_Cup[[1,4,3,2],[1,4,3,2]]=kron(sm,sp);
    Cdagdn_Cup=TensorMap(Cdagdn_Cup,  V ← V);


    sx=Cdagup_Cdn+Cdagdn_Cup;
    sy=-im*Cdagup_Cdn+im*Cdagdn_Cup;
    sz=Cdagup_Cup-Cdagdn_Cdn;


    return sx,sy,sz

#Ident, N_occu, n_double, Cdag, C=Hamiltonians_spinful_Z2(config_kwargs)




# CTM0=None;
# Fermionic_CTMRG_cell_iPESS(B_set,T_set,init,CTM0, ls_ctm_args, global_args);







        

