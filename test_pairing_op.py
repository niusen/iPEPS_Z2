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



def twosite_pairing_spinful_Z2(config_kwargs):#superconductivity order parameter
    config_Z2 = yastn.make_config(sym='Z2',fermionic=True, **config_kwargs)
    Device=config_kwargs['default_device'];

    Vp = yastn.Leg(config_Z2, s=1, t=(0, 1), D=(2, 2))
    Vp_conj = yastn.Leg(config_Z2, s=-1, t=(0, 1), D=(2, 2))
    Vdummy = yastn.Leg(config_Z2, s=-1, t=[1], D=[2])
    Vdummy_conj = yastn.Leg(config_Z2, s=1, t=[1], D=[2])

    #order of kron() command: (0,0), (0,1), (1,0), (1,1)
    order=(1-1,4-1,3-1,2-1);
    
    Id=torch.tensor([[1.0, 0], [0, 1.0]]).to(device=Device);
    sm=torch.tensor([[0, 1.0], [0, 0]]).to(device=Device); 
    sp=torch.tensor([[0, 0], [1.0, 0]]).to(device=Device);
    sz=torch.tensor([[1.0, 0], [0, -1.0]]).to(device=Device); 
    occu=torch.tensor([[0, 0], [0, 1.0]]).to(device=Device);

    Cdagup=torch.zeros((4,4,2),dtype=Id.dtype,device=Id.device);
    Cdagup[:,:,0]=torch.kron(sp,Id);
    Cdagdn=torch.zeros((4,4,2),dtype=Id.dtype,device=Id.device);
    Cdagdn[:,:,1]=torch.kron(sz,sp);
    Cdaga=Cdagup+Cdagdn
    Cdaga=Cdaga[order,:,:]
    Cdaga=Cdaga[:,order,:]
    Cdaga_empty=yastn.zeros(config=config_Z2, legs=[Vp,Vp_conj,Vdummy])
    Cdaga=fill_Z2_Ham(Cdaga,Cdaga_empty)
    Cdaga=yastn.transpose(Cdaga,axes=(2,0,1))

    Cdagup=torch.zeros((2,4,4),dtype=Id.dtype,device=Id.device);
    Cdagup[1,:,:]=torch.kron(sp,Id);
    Cdagdn=torch.zeros((2,4,4),dtype=Id.dtype,device=Id.device);
    Cdagdn[0,:,:]=torch.kron(sz,sp);
    Cdagb=Cdagup-Cdagdn;
    Cdagb=Cdagb[:,order,:]
    Cdagb=Cdagb[:,:,order]
    Cdagb_empty=yastn.zeros(config=config_Z2, legs=[Vdummy_conj, Vp,Vp_conj])
    Cdagb=fill_Z2_Ham(Cdagb,Cdagb_empty)



    # # singlet pairing
    # Cdagupa=zeros(4,4,2);
    # Cdagupa[[1,4,3,2],[1,4,3,2],1]=kron(sp,Id);
    # Cdagdna=zeros(4,4,2);
    # Cdagdna[[1,4,3,2],[1,4,3,2],2]=kron(sz,sp);
    # Cdaga=TensorMap(Cdagupa+Cdagdna,  V ← V ⊗Vdummy);
    # Cdaga=permute(Cdaga,(3,1,),(2,))

    # Cdagupb=zeros(2,4,4);
    # Cdagupb[2,[1,4,3,2],[1,4,3,2]]=kron(sp,Id);
    # Cdagdnb=zeros(2,4,4);
    # Cdagdnb[1,[1,4,3,2],[1,4,3,2]]=kron(sz,sp);
    # Cdagb=TensorMap(Cdagupb-Cdagdnb, Vdummy ⊗ V ← V);



    # @tensor pairing[:]:=Cdaga[1,-1,-3]*Cdagb[1,-2,-4];


    pairing=yastn.ncon([Cdaga,Cdagb], [[1,-1,-3], [1,-2,-4]]);

    pairing_singlet_site1=Cdaga;
    pairing_singlet_site2=Cdagb;

    pairing_string = yastn.eye(config=config_Z2,legs=Cdagb.get_legs(axes=1-1), isdiag=False)


    return pairing_singlet_site1, pairing_singlet_site2, pairing_string


twosite_pairing_spinful_Z2(config_kwargs)