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

def fill_Z2_Ham(T_dense,T):
    T=T*0;
    legs=T.get_legs();
    if T.ndim==2:
        even_dims=[legs[0].D[0], legs[1].D[0]]
        odd_dims=[legs[0].D[1], legs[1].D[1]]
        for c1 in range(0,2):
            if c1==0:
                range1=range(0,even_dims[0])
            else:
                range1=range(even_dims[0],even_dims[0]+odd_dims[0])
            for c2 in range(0,2):
                if c2==0:
                    range2=range(0,even_dims[1])
                else:
                    range2=range(even_dims[1],even_dims[1]+odd_dims[1])
                T_=T_dense[range1,:]
                T_=T_[:,range2]

                if numpy.mod(c1+c2,2)==0:
                    T.set_block(ts=(c1, c2), val=T_, Ds=(len(range1), len(range2)))
    elif T.ndim==3:
        if (len(legs[0].D)==2)&(len(legs[1].D)==2)&(len(legs[2].D)==2):
            even_dims=[legs[0].D[0], legs[1].D[0], legs[2].D[0]]
            odd_dims=[legs[0].D[1], legs[1].D[1], legs[2].D[1]]
        elif (len(legs[0].D)==1)&(legs[0].t==((1,),)):#only odd parity
            even_dims=[0, legs[1].D[0], legs[2].D[0]]
            odd_dims=[legs[0].D[1-1], legs[1].D[1], legs[2].D[1]]
        elif (len(legs[2].D)==1)&(legs[2].t==((1,),)):#only odd parity
            even_dims=[legs[0].D[0], legs[1].D[0], 0]
            odd_dims=[legs[0].D[1], legs[1].D[1], legs[2].D[1-1]]
        for c1 in range(0,2):
            if (c1==0)&(even_dims[0]>0):
                range1=range(0,even_dims[0])
            else:
                range1=range(even_dims[0],even_dims[0]+odd_dims[0])
            for c2 in range(0,2):
                if c2==0:
                    range2=range(0,even_dims[1])
                else:
                    range2=range(even_dims[1],even_dims[1]+odd_dims[1])
                for c3 in range(0,2):
                    if (c3==0)&(even_dims[2]>0):
                        range3=range(0,even_dims[2])
                    else:
                        range3=range(even_dims[2],even_dims[2]+odd_dims[2])

                    T_=T_dense[range1,:,:]
                    T_=T_[:,range2,:]
                    T_=T_[:,:,range3]

                    if numpy.mod(c1+c2+c3,2)==0:
                        T.set_block(ts=(c1, c2, c3), val=T_, Ds=(len(range1), len(range2), len(range3)))
    elif T.ndim==4:
        even_dims=[legs[0].D[0], legs[1].D[0], legs[2].D[0], legs[3].D[0]]
        odd_dims=[legs[0].D[1], legs[1].D[1], legs[2].D[1], legs[3].D[1]]
        for c1 in range(0,2):
            if c1==0:
                range1=range(0,even_dims[0])
            else:
                range1=range(even_dims[0],even_dims[0]+odd_dims[0])
            for c2 in range(0,2):
                if c2==0:
                    range2=range(0,even_dims[1])
                else:
                    range2=range(even_dims[1],even_dims[1]+odd_dims[1])
                for c3 in range(0,2):
                    if c3==0:
                        range3=range(0,even_dims[2])
                    else:
                        range3=range(even_dims[2],even_dims[2]+odd_dims[2])
                    for c4 in range(0,2):
                        if c4==0:
                            range4=range(0,even_dims[3])
                        else:
                            range4=range(even_dims[3],even_dims[3]+odd_dims[3])

                        T_=T_dense[range1,:,:,:]
                        T_=T_[:,range2,:,:]
                        T_=T_[:,:,range3,:]
                        T_=T_[:,:,:,range4]

                        if numpy.mod(c1+c2+c3+c4,2)==0:
                            T.set_block(ts=(c1, c2, c3, c4), val=T_, Ds=(len(range1), len(range2), len(range3), len(range4)))
    return T
def Hamiltonians_spinful_Z2(config_kwargs):
    config_Z2 = yastn.make_config(sym='Z2',fermionic=True, **config_kwargs)
    Device=config_kwargs['default_device'];

    Vp = yastn.Leg(config_Z2, s=1, t=(0, 1), D=(2, 2))
    Vp_conj = yastn.Leg(config_Z2, s=-1, t=(0, 1), D=(2, 2))
    Vdummy = yastn.Leg(config_Z2, s=-1, t=[1], D=[2])
    Vdummy_conj = yastn.Leg(config_Z2, s=1, t=[1], D=[2])

    

    # Vdummy=Rep[ℤ₂](1=>2);
    # V=Rep[ℤ₂](0=>2,1=>2);


    Id=torch.tensor([[1.0, 0], [0, 1.0]]).to(device=Device);
    sm=torch.tensor([[0, 1.0], [0, 0]]).to(device=Device); 
    sp=torch.tensor([[0, 0], [1.0, 0]]).to(device=Device);
    sz=torch.tensor([[1.0, 0], [0, -1.0]]).to(device=Device); 
    occu=torch.tensor([[0, 0], [0, 1.0]]).to(device=Device);
    
    #order of kron() command: (0,0), (0,1), (1,0), (1,1)
    order=(1-1,4-1,3-1,2-1);



    Ident=torch.kron(Id,Id);
    Ident=Ident[order,:];
    Ident=Ident[:,order];
    Ident_=yastn.zeros(config=config_Z2, legs=[Vp,Vp_conj])
    Ident_=fill_Z2_Ham(Ident,Ident_)

    N_occu=torch.kron(occu,Id)+torch.kron(Id,occu);
    N_occu=N_occu[order,:]
    N_occu=N_occu[:,order]
    N_occu_=yastn.zeros(config=config_Z2, legs=[Vp,Vp_conj])
    N_occu_=fill_Z2_Ham(N_occu,N_occu_)

    n_double=torch.kron(occu,occu)
    n_double=n_double[order,:]
    n_double=n_double[:,order]
    n_double_=yastn.zeros(config=config_Z2, legs=[Vp,Vp_conj])
    n_double_=fill_Z2_Ham(n_double,n_double_)


    Cdagup=torch.zeros((4,4,2),dtype=Id.dtype,device=Id.device);
    Cdagup[:,:,0]=torch.kron(sp,Id);
    Cdagdn=torch.zeros((4,4,2),dtype=Id.dtype,device=Id.device);
    Cdagdn[:,:,1]=torch.kron(sz,sp);
    # Cdag=TensorMap(,  V ← V ⊗Vdummy);
    Cdag=Cdagup+Cdagdn
    Cdag=Cdag[order,:,:]
    Cdag=Cdag[:,order,:]
    Cdag_=yastn.zeros(config=config_Z2, legs=[Vp,Vp_conj,Vdummy])
    Cdag_=fill_Z2_Ham(Cdag,Cdag_)
    Cdag_=yastn.transpose(Cdag_,axes=(2,0,1))

    Cup=torch.zeros((2,4,4),dtype=Id.dtype,device=Id.device);
    Cup[0,:,:]=torch.kron(sm,Id);
    Cdn=torch.zeros((2,4,4),dtype=Id.dtype,device=Id.device);
    Cdn[1,:,:]=torch.kron(sz,sm);
    C=Cup+Cdn;
    C=C[:,order,:]
    C=C[:,:,order]
    C_=yastn.zeros(config=config_Z2, legs=[Vdummy_conj, Vp,Vp_conj])
    C_=fill_Z2_Ham(C,C_)
    # C=TensorMap(, Vdummy ⊗ V ← V);
   
    return Ident_, N_occu_, n_double_, Cdag_, C_

Ident, N_occu, n_double, Cdag, C=Hamiltonians_spinful_Z2(config_kwargs)

print(Ident.to_dense())
print(N_occu.to_dense())
print(n_double.to_dense())
print(Cdag.to_dense())
print(C.to_dense())

# CTM0=None;
# Fermionic_CTMRG_cell_iPESS(B_set,T_set,init,CTM0, ls_ctm_args, global_args);







        

