import numpy,torch,math,cmath
import yastn
from config.settings import *
import copy
from ctmrg.Fermionic_CTMRG_unitcell_iPESS import build_double_layer_swap_Tm,build_double_layer_swap_Bm
from torch.utils.checkpoint import checkpoint

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
            elif (c1==1):
                range1=range(even_dims[0],even_dims[0]+odd_dims[0])
            else:
                continue
            for c2 in range(0,2):
                if c2==0:
                    range2=range(0,even_dims[1])
                else:
                    range2=range(even_dims[1],even_dims[1]+odd_dims[1])
                for c3 in range(0,2):
                    if (c3==0)&(even_dims[2]>0):
                        range3=range(0,even_dims[2])
                    elif (c3==1):
                        range3=range(even_dims[2],even_dims[2]+odd_dims[2])
                    else:
                        continue

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
   
    CdagC_string = yastn.eye(config=config_Z2,legs=C_.get_legs(axes=1-1), isdiag=False)
    return Ident_, N_occu_, n_double_, Cdag_, C_, CdagC_string


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
    Cdagup_Cup=torch.kron(torch.matmul(sp,sm),Id);
    Cdagup_Cup=Cdagup_Cup[order,:]
    Cdagup_Cup=Cdagup_Cup[:,order]
    Cdagup_Cup_=yastn.zeros(config=config_Z2, legs=[Vp,Vp_conj])
    Cdagup_Cup_=fill_Z2_Ham(Cdagup_Cup,Cdagup_Cup_)

    Cdagdn_Cdn=torch.zeros((4,4),dtype=Id.dtype,device=Id.device);
    Cdagdn_Cdn=torch.kron(Id,torch.matmul(sp,sm));
    Cdagdn_Cdn=Cdagdn_Cdn[order,:]
    Cdagdn_Cdn=Cdagdn_Cdn[:,order]
    Cdagdn_Cdn_=yastn.zeros(config=config_Z2, legs=[Vp,Vp_conj])
    Cdagdn_Cdn_=fill_Z2_Ham(Cdagdn_Cdn,Cdagdn_Cdn_)

    Cdagup_Cdn=torch.zeros((4,4),dtype=Id.dtype,device=Id.device);
    Cdagup_Cdn=torch.kron(sp,sm);
    Cdagup_Cdn=Cdagup_Cdn[order,:]
    Cdagup_Cdn=Cdagup_Cdn[:,order]
    Cdagup_Cdn_=yastn.zeros(config=config_Z2, legs=[Vp,Vp_conj])
    Cdagup_Cdn_=fill_Z2_Ham(Cdagup_Cdn,Cdagup_Cdn_)

    Cdagdn_Cup=torch.zeros((4,4),dtype=Id.dtype,device=Id.device);
    Cdagdn_Cup=torch.kron(sm,sp);
    Cdagdn_Cup=Cdagdn_Cup[order,:]
    Cdagdn_Cup=Cdagdn_Cup[:,order]
    Cdagdn_Cup_=yastn.zeros(config=config_Z2, legs=[Vp,Vp_conj])
    Cdagdn_Cup_=fill_Z2_Ham(Cdagdn_Cup,Cdagdn_Cup_)


    sx=Cdagup_Cdn_+Cdagdn_Cup_;
    sy=-1j*Cdagup_Cdn_+1j*Cdagdn_Cup_;
    sz=Cdagup_Cup_-Cdagdn_Cdn_;

    return sx,sy,sz

def Operators_spinful_Z2(config_kwargs):
    config_Z2 = yastn.make_config(sym='Z2',fermionic=True, **config_kwargs)
    Device=config_kwargs['default_device'];

    Vp = yastn.Leg(config_Z2, s=1, t=(0, 1), D=(2, 2))
    Vp_conj = yastn.Leg(config_Z2, s=-1, t=(0, 1), D=(2, 2))
    Vdummy = yastn.Leg(config_Z2, s=-1, t=[1], D=[2])
    Vdummy_conj = yastn.Leg(config_Z2, s=1, t=[1], D=[2])


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

 

    Sp=torch.kron(sp,sm);
    Sp=Sp[order,:]
    Sp=Sp[:,order]
    Sp_=yastn.zeros(config=config_Z2, legs=[Vp,Vp_conj])
    Sp_=fill_Z2_Ham(Sp,Sp_)

    Sm=torch.kron(sm,sp);
    Sm=Sm[order,:]
    Sm=Sm[:,order]
    Sm_=yastn.zeros(config=config_Z2, legs=[Vp,Vp_conj])
    Sm_=fill_Z2_Ham(Sm,Sm_)

    Sz=torch.kron(occu,Id)/2-torch.kron(Id,occu)/2;
    Sz=Sz[order,:]
    Sz=Sz[:,order]
    Sz_=yastn.zeros(config=config_Z2, legs=[Vp,Vp_conj])
    Sz_=fill_Z2_Ham(Sz,Sz_)

    SpSm = yastn.ncon([Sp_, Sm_], [[-1,-2], [-3,-4]]);
    SmSp = yastn.ncon([Sm_, Sp_], [[-1,-2], [-3,-4]]);
    SzSz = yastn.ncon([Sz_, Sz_], [[-1,-2], [-3,-4]]);

    SS=SpSm/2+SmSp/2+SzSz;#s1' s1 s2' s2
    # SS=yastn.transpose(SS, axes=(1-1,3-1,2-1,4-1))
    u0,s0,v0 = yastn.linalg.svd_with_truncation(SS, axes=((0, 1), (2, 3)), svd_on_cpu=True, tol=1e-8);
    SS_trun=yastn.ncon([u0,s0,v0], [[-1,-2,1],[1,2], [2,-3,-4]]);
    assert yastn.linalg.norm(SS_trun-SS)<1e-14;
    
    # Sa=permute(u0*s0,(3,1,),(2,));
    # Sb=permute(v0,(1,2,),(3,));
    Sa=yastn.ncon([u0,s0], [[-2,-3,1],[1,-1]]);
    Sb=v0
    SS_string = yastn.eye(config=config_Z2,legs=Sb.get_legs(axes=1-1), isdiag=False)

    Sx=(Sp_+Sm_)/2;
    Sy=(Sp_-Sm_)/(2*1j);
    #Hchiral[:]:=Sx[-1,-4]*Sy[-2,-5]*Sz[-3,-6]-Sx[-1,-4]*Sz[-2,-5]*Sy[-3,-6]+Sy[-1,-4]*Sz[-2,-5]*Sx[-3,-6]-Sy[-1,-4]*Sx[-2,-5]*Sz[-3,-6]+Sz[-1,-4]*Sx[-2,-5]*Sy[-3,-6]-Sz[-1,-4]*Sy[-2,-5]*Sx[-3,-6];
    xyz=yastn.ncon([Sx,Sy,Sz_], [[-1,-4],[-2,-5], [-3,-6]]);
    xzy=yastn.ncon([Sx,Sz_,Sy], [[-1,-4],[-2,-5], [-3,-6]]);
    yzx=yastn.ncon([Sy,Sz_,Sx], [[-1,-4],[-2,-5], [-3,-6]]);
    yxz=yastn.ncon([Sy,Sx,Sz_], [[-1,-4],[-2,-5], [-3,-6]]);
    zxy=yastn.ncon([Sz_,Sx,Sy], [[-1,-4],[-2,-5], [-3,-6]]);
    zyx=yastn.ncon([Sz_,Sy,Sx], [[-1,-4],[-2,-5], [-3,-6]]);
    Hchiral=xyz-xzy+yzx-yxz+zxy-zyx;
    #u,s,v=tsvd(permute(Hchiral,(1,4,),(2,3,5,6)); trunc=truncerr(1e-12));
    u,s,v=yastn.linalg.svd_with_truncation(Hchiral, axes=((1-1, 4-1), (2-1, 3-1, 5-1, 6-1)), svd_on_cpu=True, tol=1e-8);
    
    chirality_S1=u;
    # S2S3=s*v;
    S2S3=yastn.ncon([s,v], [[-1,1], [1,-2,-3,-4,-5]]);
    #u,s,v=tsvd(permute(S2S3,(1,2,4,),(3,5)); trunc=truncerr(1e-12));
    u,s,v=yastn.linalg.svd_with_truncation(S2S3, axes=((1-1, 2-1, 4-1), (3-1, 5-1)), svd_on_cpu=True, tol=1e-8);
    chirality_S2=u;
    # chirality_S3=s*v;
    chirality_S3=yastn.ncon([s,v], [[-1,1], [1,-2,-3]]);
    #@tensor Hchiral_[:]:=chirality_S1[-1,-4,1]*chirality_S2[1,-2,-5,2]*chirality_S3[2,-3,-6];
    #Hchiral_=permute(Hchiral_,(1,2,3,),(4,5,6,));
    Hchiral_trun=yastn.ncon([chirality_S1,chirality_S2,chirality_S3], [[-1,-4,1], [1,-2,-5,2], [2,-3,-6]]);
    assert yastn.linalg.norm(Hchiral-Hchiral_trun)/yastn.linalg.norm(Hchiral)<1e-12;
    chirality_string12 = yastn.eye(config=config_Z2,legs=chirality_S2.get_legs(axes=1-1), isdiag=False)
    chirality_string23 = yastn.eye(config=config_Z2,legs=chirality_S3.get_legs(axes=1-1), isdiag=False)

    return Sa, Sb, SS_string, chirality_S1,chirality_S2,chirality_S3, chirality_string12, chirality_string23





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




def build_MM_LU(Cset,Tset,AA_LU_,cx,cy,Lx,Ly):
    #@tensor MM_LU[:]:=Cset[mod1(cx,Lx)][mod1(cy,Ly)].C1[1,2]*Tset[mod1(cx+1,Lx)][mod1(cy,Ly)].T1[2,3,-3]*Tset[mod1(cx,Lx)][mod1(cy+1,Ly)].T4[-1,4,1]*AA_LU_[4,-2,-4,3]; 
    # MM_LU=permute(MM_LU,(1,2,),(3,4,));
    MM_LU=yastn.ncon([Cset[str(mod1(cx,Lx))+','+str(mod1(cy,Ly))]['C1'], Tset[str(mod1(cx+1,Lx))+','+str(mod1(cy,Ly))]['T1'], Tset[str(mod1(cx,Lx))+','+str(mod1(cy+1,Ly))]['T4'], AA_LU_], [[1,2], [2,3,-3], [-1,4,1], [4,-2,-4,3]]);
    return MM_LU


def build_MM_RU(Cset,Tset,AA_RU_,cx,cy,Lx,Ly):
    # @tensor MM_RU[:]:=Tset[mod1(cx+2,Lx)][mod1(cy,Ly)].T1[-1,3,1]* Cset[mod1(cx+3,Lx)][mod1(cy,Ly)].C2[1,2]* AA_RU_[-2,-4,4,3]* Tset[mod1(cx+3,Lx)][mod1(cy+1,Ly)].T2[2,4,-3];
    # MM_RU=permute(MM_RU,(1,2,),(3,4,));
    MM_RU=yastn.ncon([Tset[str(mod1(cx+2,Lx))+','+str(mod1(cy,Ly))]['T1'],  Cset[str(mod1(cx+3,Lx))+','+str(mod1(cy,Ly))]['C2'], AA_RU_, Tset[str(mod1(cx+3,Lx))+','+str(mod1(cy+1,Ly))]['T2']], [[-1,3,1], [1,2], [-2,-4,4,3], [2,4,-3]]);
    return MM_RU

def build_MM_LD(Cset,Tset,AA_LD_,cx,cy,Lx,Ly):
    # @tensor MM_LD[:]:=Tset[mod1(cx,Lx)][mod1(cy+2,Ly)].T4[1,3,-2]*AA_LD_[3,4,-5,-3]*Cset[mod1(cx,Lx)][mod1(cy+3,Ly)].C4[2,1]*Tset[mod1(cx+1,Lx)][mod1(cy+3,Ly)].T3[-4,4,2]; 
    # MM_LD=permute(MM_LD,(1,2,),(3,4,));
    MM_LD=yastn.ncon([Tset[str(mod1(cx,Lx))+','+str(mod1(cy+2,Ly))]['T4'], AA_LD_, Cset[str(mod1(cx,Lx))+','+str(mod1(cy+3,Ly))]['C4'], Tset[str(mod1(cx+1,Lx))+','+str(mod1(cy+3,Ly))]['T3']], [[1,3,-2], [3,4,-5,-3], [2,1], [-4,4,2]]);
    return MM_LD

def build_MM_RD(Cset,Tset,AA_RD_,cx,cy,Lx,Ly):
    # @tensor MM_RD[:]:=Tset[mod1(cx+3,Lx)][mod1(cy+2,Ly)].T2[-4,-3,2]*Tset[mod1(cx+2,Lx)][mod1(cy+3,Ly)].T3[1,-2,-1]*Cset[mod1(cx+3,Lx)][mod1(cy+3,Ly)].C3[2,1]; 
    # @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD_[-2,1,2,-4]; 
    # MM_RD=permute(MM_RD,(1,2,),(3,4,));
    MM_RD=yastn.ncon([Tset[str(mod1(cx+3,Lx))+','+str(mod1(cy+2,Ly))]['T2'], Tset[str(mod1(cx+2,Lx))+','+str(mod1(cy+3,Ly))]['T3'], Cset[str(mod1(cx+3,Lx))+','+str(mod1(cy+3,Ly))]['C3']], [[-4,-3,2], [1,-2,-1], [2,1]]);
    MM_RD=yastn.ncon([MM_RD, AA_RD_], [[-1,1,2,-3], [-2,1,2,-4]]);
    return MM_RD



def ob_2x2_iPESS(CTM,AA_LU_,AA_RU_,AA_LD_,AA_RD_,cx,cy,Lx,Ly):
    Cset=CTM['Cset'];
    Tset=CTM['Tset'];

    MM_LU=build_MM_LU(Cset,Tset,AA_LU_,cx,cy,Lx,Ly);
    MM_RU=build_MM_RU(Cset,Tset,AA_RU_,cx,cy,Lx,Ly);
    MM_LD=build_MM_LD(Cset,Tset,AA_LD_,cx,cy,Lx,Ly);
    MM_RD=build_MM_RD(Cset,Tset,AA_RD_,cx,cy,Lx,Ly);

    # M1=MM_LU*MM_RU;
    # M2=MM_LD*MM_RD;
    # rho=@tensor M1[1,2,3,4,]*M2[1,2,3,4];

    rho=yastn.ncon([MM_LU, MM_RU, MM_LD, MM_RD], [[5,6,1,2], [1,2,7,8], [5,6,3,4], [3,4,7,8]]);
    return rho




def get_AA_simple(double_B_set,double_T_set,pos):
    # @tensor AA[:]:=double_B_set[pos[1]][pos[2]][-1,1,-4]*double_T_set[pos[1]][pos[2]][-2,-3,1];
    AA=yastn.ncon([double_B_set[str(pos[1-1])+','+str(pos[2-1])], double_T_set[str(pos[1-1])+','+str(pos[2-1])]], [[-1,1,-4], [-2,-3,1]]);
    return AA




def ob_onsite_iPESS(CTM,O1,B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly):
 
    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];


    # @tensor A1[:]:= A_cell[pos_LU[1-1]][pos_LU[2-1]][-1,-2,-3,-4,1]*O1[-5,1]

    B_LU=B_set[str(pos_LU[1-1])+','+str(pos_LU[2-1])];#(LU,M)
    T_LU=T_set[str(pos_LU[1-1])+','+str(pos_LU[2-1])];#(M,dRD)
    B_LU0=B_LU.clone();
    T_LU0=T_LU.clone();

    # @tensor T_LU[:]:=T_LU[-1,1,-3,-4]*O1[-2,1];#M,d,R,D
    T_LU=yastn.ncon([T_LU,O1], [[-1,1,-3,-4], [-2,1]]);

    # B_LU=permute(B_LU,(1,2,),(3,));
    # T_LU=permute(T_LU,(1,),(2,3,4,));
    B_LU_double= build_double_layer_swap_Tm(B_LU0.conj(),B_LU, False);#L M U
    T_LU_double= build_double_layer_swap_Bm(T_LU0.conj(),T_LU, True);#D R M

    # @tensor AA_LU[:]:=B_LU_double[-1,1,-4]*T_LU_double[-2,-3,1];
    AA_LU=yastn.ncon([B_LU_double,T_LU_double], [[-1,1,-4], [-2,-3,1]]);
    ##############################
    AA_LU0=get_AA_simple(double_B_set,double_T_set,pos_LU);
    AA_LD0=get_AA_simple(double_B_set,double_T_set,pos_LD);
    AA_RU0=get_AA_simple(double_B_set,double_T_set,pos_RU);
    AA_RD0=get_AA_simple(double_B_set,double_T_set,pos_RD);

    ob=ob_2x2_iPESS(CTM,AA_LU, AA_RU0,AA_LD0,AA_RD0,cx,cy,Lx,Ly).to_number();
    Norm=ob_2x2_iPESS(CTM,AA_LU0,AA_RU0,AA_LD0,AA_RD0,cx,cy,Lx,Ly).to_number();
    ob=ob/Norm;
    return ob


def hopping_x_iPESS(CTM,O1,O2,string12, B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly):

    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

    #########################################

    B_LU=B_set[str(pos_LU[1-1])+','+str(pos_LU[2-1])];#(LU,M)
    T_LU=T_set[str(pos_LU[1-1])+','+str(pos_LU[2-1])];#(M,dRD)
    B_LU0=B_LU.clone();
    T_LU0=T_LU.clone();

    #@tensor T_LU[:]:=T_LU[-1,1,-3,-4]*O1[-5,-2,1];#M,d,R,D,virtual
    T_LU=yastn.ncon([T_LU,O1], [[-1,1,-3,-4], [-5,-2,1]]);
    # gate=@ignore_derivatives parity_gate(B_LU,1);#L 
    # @tensor B_LU[:]:=B_LU[1,-2,-3]*gate[-1,1];#L,U,M
    B_LU=yastn.swap_gate(B_LU, (1-1,1-1));
    # gate=@ignore_derivatives parity_gate(T_LU,4); #D
    # @tensor T_LU[:]:=T_LU[-1,-2,-3,1,-5]*gate[-4,1];#M,d,R,D,virtual
    T_LU=yastn.swap_gate(T_LU, (4-1,4-1));
    # gate=@ignore_derivatives parity_gate(B_LU,2);#U 
    # @tensor B_LU[:]:=B_LU[-1,1,-3]*gate[-2,1];#L,U,M
    B_LU=yastn.swap_gate(B_LU, (2-1,2-1));
    # U=@ignore_derivatives unitary(fuse(space(T_LU,3)⊗space(T_LU,5)), space(T_LU,3)⊗space(T_LU,5)); 
    # @tensor T_LU[:]:=T_LU[-1,-2,1,-4,2]*U[-3,1,2];#M,d,R',D
    T_LU=T_LU.fuse_legs((0,1,(2,4),3,), mode='hard');

    # B_LU=permute(B_LU,(1,2,),(3,));
    # T_LU=permute(T_LU,(1,),(2,3,4,));
    B_LU_double= build_double_layer_swap_Tm(B_LU0.conj(),B_LU, False);#L M U
    T_LU_double= build_double_layer_swap_Bm(T_LU0.conj(),T_LU, True);#D R M
    ###########################################

    B_RU=B_set[str(pos_RU[1-1])+','+str(pos_RU[2-1])];#(LU,M)
    T_RU=T_set[str(pos_RU[1-1])+','+str(pos_RU[2-1])];#(M,dRD)
    B_RU0=B_RU.clone();
    T_RU0=T_RU.clone();

    # @tensor T_RU[:]:= T_RU[-1,1,-3,-4]*O2[-5,-2,1];#M,d,R,D,virtual'
    T_RU=yastn.ncon([T_RU,O2], [[-1,1,-3,-4], [-5,-2,1]]);
    # gate=@ignore_derivatives parity_gate(B_RU,1); #L
    # @tensor B_RU[:]:=B_RU[1,-2,-3]*gate[-1,1];#L,U,M
    B_RU=yastn.swap_gate(B_RU, (1-1,1-1));
    # gate=@ignore_derivatives parity_gate(B_RU,2); #U
    # @tensor B_RU[:]:=B_RU[-1,1,-3]*gate[-2,1];#L,U,M
    B_RU=yastn.swap_gate(B_RU, (2-1,2-1));
    
    # U2=@ignore_derivatives unitary(fuse(space(T_RU,1)⊗space(T_RU,5)), space(T_RU,1)⊗space(T_RU,5));
    # @tensor T_RU[:]:=T_RU[1,-2,-3,-4,2]*U2[-1,1,2];#M',d,R,D
    T_RU=T_RU.fuse_legs(((0,4),1,2,3,), mode='hard');

    #O_string=@ignore_derivatives unitary(space(O1,1)',space(O1,1)');
    # @tensor B_RU[:]:=B_RU[-1,-2,-3]*O_string[-4,-5];#(L,U,M), (virtual,virtual')=>(L,U,M, virtual,virtual')
    B_RU=yastn.ncon([B_RU,string12], [[-1,-2,-3], [-4,-5]]);
    # @tensor B_RU[:]:=B_RU[1,-2,3,2,4]*U'[1,2,-1]*U2'[3,4,-3];#L,U,M
    B_RU=B_RU.fuse_legs(((0,3),1,(2,4)), mode='hard');


    # B_RU=permute(B_RU,(1,2,),(3,));
    # T_RU=permute(T_RU,(1,),(2,3,4,));
    B_RU_double= build_double_layer_swap_Tm(B_RU0.conj(),B_RU, False);#L M U
    T_RU_double= build_double_layer_swap_Bm(T_RU0.conj(),T_RU, True);#D R M
    ####################################
    # @tensor AA_LU[:]:=B_LU_double[-1,1,-4]*T_LU_double[-2,-3,1];
    # @tensor AA_RU[:]:=B_RU_double[-1,1,-4]*T_RU_double[-2,-3,1];
    AA_LU=yastn.ncon([B_LU_double,T_LU_double], [[-1,1,-4], [-2,-3,1]]);
    AA_RU=yastn.ncon([B_RU_double,T_RU_double], [[-1,1,-4], [-2,-3,1]]);

      

    AA_LU0=get_AA_simple(double_B_set,double_T_set,pos_LU);
    AA_LD0=get_AA_simple(double_B_set,double_T_set,pos_LD);
    AA_RU0=get_AA_simple(double_B_set,double_T_set,pos_RU);
    AA_RD0=get_AA_simple(double_B_set,double_T_set,pos_RD);

    ob=ob_2x2_iPESS(CTM,AA_LU,AA_RU,AA_LD0,AA_RD0,cx,cy,Lx,Ly).to_number();
    Norm=ob_2x2_iPESS(CTM,AA_LU0,AA_RU0,AA_LD0,AA_RD0,cx,cy,Lx,Ly).to_number();
    ob=ob/Norm;
    return ob


def hopping_y_iPESS(CTM,O1,O2,string12,B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly):

    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

    #############################################
    ####
    B_RU=B_set[str(pos_RU[1-1])+','+str(pos_RU[2-1])];#(LU,M)
    T_RU=T_set[str(pos_RU[1-1])+','+str(pos_RU[2-1])];#(M,dRD)
    B_RU0=B_RU.clone();
    T_RU0=T_RU.clone();

    # @tensor T_RU[:]:= T_RU[-1,1,-3,-4]*O1[-5,-2,1];#M,d,R,D,virtual
    T_RU=yastn.ncon([T_RU,O1], [[-1,1,-3,-4], [-5,-2,1]]);
    # gate=@ignore_derivatives parity_gate(B_RU,1); #L
    # @tensor B_RU[:]:=B_RU[1,-2,-3]*gate[-1,1];#L,U,M
    B_RU=yastn.swap_gate(B_RU, (1-1,1-1));
    # gate=@ignore_derivatives parity_gate(T_RU,4); #D
    # @tensor T_RU[:]:=T_RU[-1,-2,-3,1,-5]*gate[-4,1];#M,d,R,D,virtual
    T_RU=yastn.swap_gate(T_RU, (4-1,4-1));
    # gate=@ignore_derivatives parity_gate(B_RU,2); #U
    # @tensor B_RU[:]:=B_RU[-1,1,-3]*gate[-2,1];#L,U,M
    B_RU=yastn.swap_gate(B_RU, (2-1,2-1));
    # U1=@ignore_derivatives unitary(fuse(space(T_RU,4)⊗space(T_RU,5)), space(T_RU,4)⊗space(T_RU,5)); 
    # @tensor T_RU[:]:=T_RU[-1,-2,-3,1,2]*U1[-4,1,2];#M,d,R,D'
    T_RU=T_RU.fuse_legs((0,1,2,(3,4)), mode='hard');

    # B_RU=permute(B_RU,(1,2,),(3,));
    # T_RU=permute(T_RU,(1,),(2,3,4,));
    B_RU_double= build_double_layer_swap_Tm(B_RU0.conj(),B_RU, False);#L M U
    T_RU_double= build_double_layer_swap_Bm(T_RU0.conj(),T_RU, True);#D R M
    ####################################
    # @tensor A_RD[:]:= A_cell[pos_RD[1-1]][pos_RD[2-1]][-1,-2,-3,-4,1]*O2[-6,-5,1]
    # @tensor A_RD[:]:=A_RD[-1,-2,-3,1,-5,2]*U1'[1,2,-4];
    ####
    B_RD=B_set[str(pos_RD[1-1])+','+str(pos_RD[2-1])];#(LU,M)
    T_RD=T_set[str(pos_RD[1-1])+','+str(pos_RD[2-1])];#(M,dRD)
    B_RD0=B_RD.clone();
    T_RD0=T_RD.clone();

    
    # @tensor T_RD[:]:= T_RD[-1,1,-3,-4]*O2[-5,-2,1];#M,d,R,D,virtual
    T_RD=yastn.ncon([T_RD,O2], [[-1,1,-3,-4], [-5,-2,1]]);
    # U2=@ignore_derivatives unitary(fuse(space(T_RD,1)⊗space(T_RD,5)), space(T_RD,1)⊗space(T_RD,5));
    # @tensor T_RD[:]:=T_RD[1,-2,-3,-4,2]*U2[-1,1,2];#M',d,R,D
    T_RD=T_RD.fuse_legs(((0,4),1,2,3), mode='hard');

    # O_string=@ignore_derivatives unitary(space(O1,1)',space(O1,1)');
    # @tensor B_RD[:]:= B_RD[-1,-2,-3]*O_string[-4,-5];#(L,U,M), (virtual',virtual)=>(L,U,M, virtual',virtual)
    B_RD=yastn.ncon([B_RD,string12], [[-1,-2,-3], [-4,-5]]);
    # @tensor B_RD[:]:=B_RD[-1,1,3,2,4]*U1'[1,2,-2]*U2'[3,4,-3];#L,U,M
    B_RD=B_RD.fuse_legs((0,(1,3),(2,4)), mode='hard');

    # B_RD=permute(B_RD,(1,2,),(3,));
    # T_RD=permute(T_RD,(1,),(2,3,4,));

    B_RD_double= build_double_layer_swap_Tm(B_RD0.conj(),B_RD, False);#L M U
    T_RD_double= build_double_layer_swap_Bm(T_RD0.conj(),T_RD, True);#D R M
    ###################################################
    # @tensor AA_RU[:]:=B_RU_double[-1,1,-4]*T_RU_double[-2,-3,1];
    # @tensor AA_RD[:]:=B_RD_double[-1,1,-4]*T_RD_double[-2,-3,1];
    AA_RU=yastn.ncon([B_RU_double,T_RU_double], [[-1,1,-4], [-2,-3,1]]);
    AA_RD=yastn.ncon([B_RD_double,T_RD_double], [[-1,1,-4], [-2,-3,1]]);

    AA_LU0=get_AA_simple(double_B_set,double_T_set,pos_LU);
    AA_LD0=get_AA_simple(double_B_set,double_T_set,pos_LD);
    AA_RU0=get_AA_simple(double_B_set,double_T_set,pos_RU);
    AA_RD0=get_AA_simple(double_B_set,double_T_set,pos_RD);
    
    ob=ob_2x2_iPESS(CTM,AA_LU0,AA_RU,AA_LD0,AA_RD,cx,cy,Lx,Ly).to_number();
    Norm=ob_2x2_iPESS(CTM,AA_LU0,AA_RU0,AA_LD0,AA_RD0,cx,cy,Lx,Ly).to_number();
    ob=ob/Norm;
    return ob


def hopping_diagonala_iPESS(CTM,O1,O2,string12,B_set,T_set, double_B_set, double_T_set, cx,cy,Lx,Ly):

    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

    ###################################################
    ######
    B_LD=B_set[str(pos_LD[1-1])+','+str(pos_LD[2-1])];#(LU,M)
    T_LD=T_set[str(pos_LD[1-1])+','+str(pos_LD[2-1])];#(M,dRD)
    B_LD0=B_LD.clone();
    T_LD0=T_LD.clone();

    # @tensor T_LD[:]:= T_LD[-1,1,-2,-3]*O1[-5,-4,1];#M,R,D,d,virtual
    T_LD=yastn.ncon([T_LD,O1], [[-1,1,-2,-3], [-5,-4,1]]);
    # gate=@ignore_derivatives parity_gate(B_LD,1); #L
    # @tensor B_LD[:]:=B_LD[1,-2,-3]*gate[-1,1];#L,U,M
    B_LD=yastn.swap_gate(B_LD, (1-1,1-1));
    # gate=@ignore_derivatives parity_gate(T_LD,3); #D
    # @tensor T_LD[:]:=T_LD[-1,-2,1,-4,-5]*gate[-3,1];#M,R,D,d,virtual
    T_LD=yastn.swap_gate(T_LD, (3-1,3-1));
    # gate=@ignore_derivatives parity_gate(B_LD,2); #U
    # @tensor B_LD[:]:=B_LD[-1,1,-3]*gate[-2,1];#L,U,M
    B_LD=yastn.swap_gate(B_LD, (2-1,2-1));
    # U1=@ignore_derivatives unitary(fuse(space(T_LD,2)⊗space(T_LD,5)), space(T_LD,2)⊗space(T_LD,5)); 
    # @tensor T_LD[:]:=T_LD[-1,1,-3,-4,2]*U1[-2,1,2];#M,R',D,d
    T_LD=T_LD.fuse_legs((0,(1,4),2,3), mode='hard');
    # T_LD=permute(T_LD,(1,4,2,3,));#M,d,R',D
    T_LD=yastn.transpose(T_LD, axes=(1-1,4-1,2-1,3-1));

    # B_LD=permute(B_LD,(1,2,),(3,));
    # T_LD=permute(T_LD,(1,),(2,3,4,));

    B_LD_double= build_double_layer_swap_Tm(B_LD0.conj(),B_LD, False);#L M U
    T_LD_double= build_double_layer_swap_Bm(T_LD0.conj(),T_LD, True);#D R M

    #############################################
    ######
    B_RU=B_set[str(pos_RU[1-1])+','+str(pos_RU[2-1])];#(LU,M)
    T_RU=T_set[str(pos_RU[1-1])+','+str(pos_RU[2-1])];#(M,dRD)
    B_RU0=B_RU.clone();
    T_RU0=T_RU.clone();

    # @tensor T_RU[:]:= T_RU[-1,1,-2,-3]*O2[-5,-4,1];#M,R,D,d,virtual
    T_RU=yastn.ncon([T_RU,O2], [[-1,1,-2,-3], [-5,-4,1]]);
    # gate=@ignore_derivatives parity_gate(T_RU,2); #R
    # @tensor T_RU[:]:=T_RU[-1,1,-3,-4,-5]*gate[-2,1];
    T_RU=yastn.swap_gate(T_RU, (2-1,2-1));
    # gate=@ignore_derivatives parity_gate(T_RU,4); #d
    # @tensor T_RU[:]:=T_RU[-1,-2,-3,1,-5]*gate[-4,1];
    T_RU=yastn.swap_gate(T_RU, (4-1,4-1));
    # U2=@ignore_derivatives unitary(fuse(space(T_RU,3)⊗space(T_RU,5)), space(T_RU,3)⊗space(T_RU,5)); 
    # @tensor T_RU[:]:=T_RU[-1,-2,1,-4,2]*U2[-3,1,2];#M,R,D',d
    T_RU=T_RU.fuse_legs((0,1,(2,4),3), mode='hard');
    # T_RU=permute(T_RU,(1,4,2,3,));#M,d,R,D
    T_RU=yastn.transpose(T_RU, axes=(1-1,4-1,2-1,3-1));

    # B_RU=permute(B_RU,(1,2,),(3,));
    # T_RU=permute(T_RU,(1,),(2,3,4,));

    B_RU_double= build_double_layer_swap_Tm(B_RU0.conj(),B_RU, False);#L M U
    T_RU_double= build_double_layer_swap_Bm(T_RU0.conj(),T_RU, True);#D R M
    ################################################
    ######
    B_RD=B_set[str(pos_RD[1-1])+','+str(pos_RD[2-1])];#(LU,M)
    T_RD=T_set[str(pos_RD[1-1])+','+str(pos_RD[2-1])];#(M,dRD)
    B_RD0=B_RD.clone();
    T_RD0=T_RD.clone();

    
    # gate=@ignore_derivatives parity_gate(T_RD,4); # D
    # @tensor T_RD[:]:=T_RD[-1,-2,-3,1]*gate[-4,1];#M,d,R,D
    T_RD=yastn.swap_gate(T_RD, (4-1,4-1));
    # gate=@ignore_derivatives parity_gate(T_RD,3); # R
    # @tensor T_RD[:]:=T_RD[-1,-2,1,-4]*gate[-3,1];#M,d,R,D
    T_RD=yastn.swap_gate(T_RD, (3-1,3-1));
    # gate=@ignore_derivatives parity_gate(T_RD,2); # d
    # @tensor T_RD[:]:=T_RD[-1,1,-3,-4]*gate[-2,1];#M,d,R,D
    T_RD=yastn.swap_gate(T_RD, (2-1,2-1));
    # O_string=@ignore_derivatives unitary(space(O1,1),space(O1,1));
    # @tensor B_RD[:]:=B_RD[1,3,-3]*O_string[4,2]*U1'[1,2,-1]*U2'[3,4,-2];#L,U,M
    B_RD=yastn.ncon([B_RD,string12], [[-1,-2,-3], [-4,-5]]);
    B_RD=B_RD.fuse_legs(((0,3),(1,4),2), mode='hard');


    # B_RD=permute(B_RD,(1,2,),(3,));
    # T_RD=permute(T_RD,(1,),(2,3,4,));

    B_RD_double= build_double_layer_swap_Tm(B_RD0.conj(),B_RD, False);#L M U
    T_RD_double= build_double_layer_swap_Bm(T_RD0.conj(),T_RD, True);#D R M
    ################################################
    # @tensor AA_LD[:]:=B_LD_double[-1,1,-4]*T_LD_double[-2,-3,1];
    # @tensor AA_RU[:]:=B_RU_double[-1,1,-4]*T_RU_double[-2,-3,1];
    # @tensor AA_RD[:]:=B_RD_double[-1,1,-4]*T_RD_double[-2,-3,1];
    AA_LD=yastn.ncon([B_LD_double,T_LD_double], [[-1,1,-4], [-2,-3,1]]);
    AA_RU=yastn.ncon([B_RU_double,T_RU_double], [[-1,1,-4], [-2,-3,1]]);
    AA_RD=yastn.ncon([B_RD_double,T_RD_double], [[-1,1,-4], [-2,-3,1]]);
    ################################################

    AA_LU0=get_AA_simple(double_B_set,double_T_set,pos_LU);
    AA_LD0=get_AA_simple(double_B_set,double_T_set,pos_LD);
    AA_RU0=get_AA_simple(double_B_set,double_T_set,pos_RU);
    AA_RD0=get_AA_simple(double_B_set,double_T_set,pos_RD);

    ob=ob_2x2_iPESS(CTM,AA_LU0,AA_RU,AA_LD,AA_RD,cx,cy,Lx,Ly).to_number();
    Norm=ob_2x2_iPESS(CTM,AA_LU0,AA_RU0,AA_LD0,AA_RD0,cx,cy,Lx,Ly).to_number();
    ob=ob/Norm;
    return ob        


# ###################################################################################
# #spin observable

def ob_dn_triangle_iPESS(CTM,S1,S2,S3,string12, string23, B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly):

    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

    ####################

    B_LD=B_set[str(pos_LD[1-1])+','+str(pos_LD[2-1])];#(LU,M)
    T_LD=T_set[str(pos_LD[1-1])+','+str(pos_LD[2-1])];#(M,dRD)
    B_LD0=B_LD.clone();
    T_LD0=T_LD.clone();


    # @tensor T_LD[:]:= T_LD[-1,1,-3,-4]*S1[-2,1,-5];#M,d,R,D,virtual
    T_LD=yastn.ncon([T_LD,S1], [[-1,1,-3,-4], [-2,1,-5]]);
    # U10=@ignore_derivatives unitary(fuse(space(T_LD,1)⊗space(T_LD,5)), space(T_LD,1)⊗space(T_LD,5)); 
    # @tensor T_LD[:]:=T_LD[1,-2,-3,-4,2]*U10[-1,1,2];#M',d,R,D
    T_LD=T_LD.fuse_legs(((0,4),1,2,3), mode='hard');

    # String1=unitary(space(S1,3),space(S1,3));
    # U1=@ignore_derivatives unitary(fuse(space(B_LD,2)⊗space(String1,1)), space(B_LD,2)⊗space(String1,1)); 
    # @tensor B_LD[:]:=B_LD[-1,1,3]*String1[2,4]*U10'[3,4,-3]*U1[-2,1,2];#L,U,M
    B_LD=yastn.ncon([B_LD,string12], [[-1,-2,-3], [-4,-5]]);
    B_LD=B_LD.fuse_legs((0,(1,4),(2,3)), mode='hard');

    # B_LD=permute(B_LD,(1,2,),(3,));
    # T_LD=permute(T_LD,(1,),(2,3,4,));

    B_LD_double= build_double_layer_swap_Tm(B_LD0.conj(),B_LD, False);#L M U
    T_LD_double= build_double_layer_swap_Bm(T_LD0.conj(),T_LD, True);#D R M
    ####################

    B_RU=B_set[str(pos_RU[1-1])+','+str(pos_RU[2-1])];#(LU,M)
    T_RU=T_set[str(pos_RU[1-1])+','+str(pos_RU[2-1])];#(M,dRD)
    B_RU0=B_RU.clone();
    T_RU0=T_RU.clone();

    # @tensor T_RU[:]:= T_RU[-1,1,-3,-4]*S3[-5,-2,1];#M,d,R,D,virtual
    T_RU=yastn.ncon([T_RU,S3], [[-1,1,-3,-4], [-5,-2,1]]);
    # U20=@ignore_derivatives unitary(fuse(space(T_RU,1)⊗space(T_RU,5)), space(T_RU,1)⊗space(T_RU,5)); 
    # @tensor T_RU[:]:=T_RU[1,-2,-3,-4,2]*U20[-1,1,2];
    T_RU=T_RU.fuse_legs(((0,4),1,2,3), mode='hard');

    # String2=unitary(space(S3,1),space(S3,1));
    # U2=@ignore_derivatives unitary(fuse(space(B_RU,1)⊗space(String2,1)), space(B_RU,1)⊗space(String2,1)); 
    # @tensor B_RU[:]:=B_RU[1,-2,3]*String2[2,4]*U2[-1,1,2]*U20'[3,4,-3];
    B_RU=yastn.ncon([B_RU,string23], [[-1,-2,-3], [-4,-5]]);
    B_RU=B_RU.fuse_legs(((0,3),1,(2,4)), mode='hard');


    # B_RU=permute(B_RU,(1,2,),(3,));
    # T_RU=permute(T_RU,(1,),(2,3,4,));

    B_RU_double= build_double_layer_swap_Tm(B_RU0.conj(),B_RU, False);#L M U
    T_RU_double= build_double_layer_swap_Bm(T_RU0.conj(),T_RU, True);#D R M
    ####################
    B_LU=B_set[str(pos_LU[1-1])+','+str(pos_LU[2-1])];#(LU,M)
    T_LU=T_set[str(pos_LU[1-1])+','+str(pos_LU[2-1])];#(M,dRD)
    B_LU0=B_LU.clone();
    T_LU0=T_LU.clone();

    # @tensor T_LU[:]:=T_LU[-1,1,2,4]*S2[5,-2,1,3]*U1'[4,5,-4]*U2'[2,3,-3];
    T_LU=yastn.ncon([T_LU,S2], [[-1,1,-3,-4], [-5,-2,1,-6]]);
    T_LU=T_LU.fuse_legs((0,1,(2,5),(3,4)), mode='hard');

    # B_LU=permute(B_LU,(1,2,),(3,));
    # T_LU=permute(T_LU,(1,),(2,3,4,));

    B_LU_double= build_double_layer_swap_Tm(B_LU0.conj(),B_LU, False);#L M U
    T_LU_double= build_double_layer_swap_Bm(T_LU0.conj(),T_LU, True);#D R M
    ####################

    # @tensor AA_LD[:]:=B_LD_double[-1,1,-4]*T_LD_double[-2,-3,1];
    # @tensor AA_RU[:]:=B_RU_double[-1,1,-4]*T_RU_double[-2,-3,1];
    # @tensor AA_LU[:]:=B_LU_double[-1,1,-4]*T_LU_double[-2,-3,1];
    AA_LD=yastn.ncon([B_LD_double,T_LD_double], [[-1,1,-4], [-2,-3,1]]);
    AA_RU=yastn.ncon([B_RU_double,T_RU_double], [[-1,1,-4], [-2,-3,1]]);
    AA_LU=yastn.ncon([B_LU_double,T_LU_double], [[-1,1,-4], [-2,-3,1]]);
    
    ################################################

    AA_LU0=get_AA_simple(double_B_set,double_T_set,pos_LU);
    AA_LD0=get_AA_simple(double_B_set,double_T_set,pos_LD);
    AA_RU0=get_AA_simple(double_B_set,double_T_set,pos_RU);
    AA_RD0=get_AA_simple(double_B_set,double_T_set,pos_RD);

    ob=ob_2x2_iPESS(CTM,AA_LU,AA_RU,AA_LD,AA_RD0,cx,cy,Lx,Ly).to_number();
    Norm=ob_2x2_iPESS(CTM,AA_LU0,AA_RU0,AA_LD0,AA_RD0,cx,cy,Lx,Ly).to_number();
    ob=ob/Norm;
    return ob        


def ob_up_triangle_iPESS(CTM,S1,S2,S3,string12, string23, B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly):

    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

    B_LD=B_set[str(pos_LD[1-1])+','+str(pos_LD[2-1])];#(LU,M)
    T_LD=T_set[str(pos_LD[1-1])+','+str(pos_LD[2-1])];#(M,dRD)
    B_LD0=B_LD.clone();
    T_LD0=T_LD.clone();

    # @tensor T_LD[:]:= T_LD[-1,1,-2,-3]*S1[-4,1,-5];#M,R,D,d,virtual
    T_LD=yastn.ncon([T_LD,S1], [[-1,1,-2,-3], [-4,1,-5]]);
    # U1=@ignore_derivatives unitary(fuse(space(T_LD,2)⊗space(T_LD,5)), space(T_LD,2)⊗space(T_LD,5)); 
    # @tensor T_LD[:]:=T_LD[-1,1,-3,-4,2]*U1[-2,1,2];#M,R',D,d
    T_LD=T_LD.fuse_legs((0,(1,4),2,3), mode='hard');
    # T_LD=permute(T_LD,(1,4,2,3,));#M,d,R',D
    T_LD=yastn.transpose(T_LD, axes=(1-1,4-1,2-1,3-1))

    # B_LD=permute(B_LD,(1,2,),(3,));
    # T_LD=permute(T_LD,(1,),(2,3,4,));

    B_LD_double= build_double_layer_swap_Tm(B_LD0.conj(),B_LD, False);#L M U
    T_LD_double= build_double_layer_swap_Bm(T_LD0.conj(),T_LD, True);#D R M

    ###################

    B_RU=B_set[str(pos_RU[1-1])+','+str(pos_RU[2-1])];#(LU,M)
    T_RU=T_set[str(pos_RU[1-1])+','+str(pos_RU[2-1])];#(M,dRD)
    B_RU0=B_RU.clone();
    T_RU0=T_RU.clone();

    # @tensor T_RU[:]:= T_RU[-1,1,-2,-3]*S3[-5,-4,1];#M,R,D,d,virtual
    T_RU=yastn.ncon([T_RU,S3], [[-1,1,-2,-3], [-5,-4,1]]);
    # U2=@ignore_derivatives unitary(fuse(space(T_RU,3)⊗space(T_RU,5)), space(T_RU,3)⊗space(T_RU,5)); 
    # @tensor T_RU[:]:=T_RU[-1,-2,1,-4,2]*U2[-3,1,2];#M,R,D',d
    T_RU=T_RU.fuse_legs((0,1,(2,4),3), mode='hard');
    # T_RU=permute(T_RU,(1,4,2,3,));#M,d,R,D
    T_RU=yastn.transpose(T_RU, axes=(1-1,4-1,2-1,3-1))

    # B_RU=permute(B_RU,(1,2,),(3,));
    # T_RU=permute(T_RU,(1,),(2,3,4,));

    B_RU_double= build_double_layer_swap_Tm(B_RU0.conj(),B_RU, False);#L M U
    T_RU_double= build_double_layer_swap_Bm(T_RU0.conj(),T_RU, True);#D R M
    ########################

    B_RD=B_set[str(pos_RD[1-1])+','+str(pos_RD[2-1])];#(LU,M)
    T_RD=T_set[str(pos_RD[1-1])+','+str(pos_RD[2-1])];#(M,dRD)
    B_RD0=B_RD.clone();
    T_RD0=T_RD.clone();

    # U_S2=@ignore_derivatives unitary(fuse(space(S2,2)*space(S2,3)), space(S2,2)*space(S2,3));
    # @tensor S2_[:]:=S2[-1,1,2,-3]*U_S2[-2,1,2];
    # U3=unitary(fuse(space(B_RD,3)*space(S2_,2)),space(B_RD,3)*space(S2_,2));
    # @tensor T_RD[:]:=T_RD[3,1,-3,-4]*U_S2'[-2,1,2]*U3'[3,2,-1];
    # @tensor B_RD[:]:=B_RD[1,3,5]*S2_[2,6,4]*U1'[1,2,-1]*U2'[3,4,-2]*U3[-3,5,6];#L,U,M
    T_RD=yastn.ncon([T_RD,S2], [[-1,1,-3,-4], [-5,-2,1,-6]]);#M,d,R,D,V1,V2
    B_RD=yastn.ncon([B_RD,string12,string23], [[-1,-2,-3], [-4,-5], [-6,-7]]);#L,U,M,V1,V1',V2',V2
    B_RD=B_RD.fuse_legs(((0,3),(1,6),(2,4,5)), mode='hard');
    T_RD=T_RD.fuse_legs(((0,4,5),1,2,3), mode='hard');


    # B_RD=permute(B_RD,(1,2,),(3,));
    # T_RD=permute(T_RD,(1,),(2,3,4,));

    B_RD_double= build_double_layer_swap_Tm(B_RD0.conj(),B_RD, False);#L M U
    T_RD_double= build_double_layer_swap_Bm(T_RD0.conj(),T_RD, True);#D R M
    ######################
    # @tensor AA_LD[:]:=B_LD_double[-1,1,-4]*T_LD_double[-2,-3,1];
    # @tensor AA_RU[:]:=B_RU_double[-1,1,-4]*T_RU_double[-2,-3,1];
    # @tensor AA_RD[:]:=B_RD_double[-1,1,-4]*T_RD_double[-2,-3,1];
    AA_LD=yastn.ncon([B_LD_double,T_LD_double], [[-1,1,-4], [-2,-3,1]]);
    AA_RU=yastn.ncon([B_RU_double,T_RU_double], [[-1,1,-4], [-2,-3,1]]);
    AA_RD=yastn.ncon([B_RD_double,T_RD_double], [[-1,1,-4], [-2,-3,1]]);
    ################################################

    AA_LU0=get_AA_simple(double_B_set,double_T_set,pos_LU);
    AA_LD0=get_AA_simple(double_B_set,double_T_set,pos_LD);
    AA_RU0=get_AA_simple(double_B_set,double_T_set,pos_RU);
    AA_RD0=get_AA_simple(double_B_set,double_T_set,pos_RD);

    ob=ob_2x2_iPESS(CTM,AA_LU0,AA_RU,AA_LD,AA_RD,cx,cy,Lx,Ly).to_number();
    Norm=ob_2x2_iPESS(CTM,AA_LU0,AA_RU0,AA_LD0,AA_RD0,cx,cy,Lx,Ly).to_number();
    ob=ob/Norm;

    return ob        




def hopping_x_iPESS_no_sign(CTM,O1,O2,string12, B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly):

    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

    #########################################


    B_LU=B_set[str(pos_LU[1-1])+','+str(pos_LU[2-1])];#(LU,M)
    T_LU=T_set[str(pos_LU[1-1])+','+str(pos_LU[2-1])];#(M,dRD)
    B_LU0=B_LU.clone();
    T_LU0=T_LU.clone();

    # @tensor T_LU[:]:=T_LU[-1,1,-3,-4]*O1[-5,-2,1];#M,d,R,D,virtual
    T_LU=yastn.ncon([T_LU,O1], [[-1,1,-3,-4], [-5,-2,1]]);
    # U=@ignore_derivatives unitary(fuse(space(T_LU,3)⊗space(T_LU,5)), space(T_LU,3)⊗space(T_LU,5)); 
    # @tensor T_LU[:]:=T_LU[-1,-2,1,-4,2]*U[-3,1,2];#M,d,R',D
    T_LU=T_LU.fuse_legs((0,1,(2,4),3,), mode='hard');

    # B_LU=permute(B_LU,(1,2,),(3,));
    # T_LU=permute(T_LU,(1,),(2,3,4,));
    B_LU_double= build_double_layer_swap_Tm(B_LU0.conj(),B_LU, False);#L M U
    T_LU_double= build_double_layer_swap_Bm(T_LU0.conj(),T_LU, True);#D R M
    ###########################################


    B_RU=B_set[str(pos_RU[1-1])+','+str(pos_RU[2-1])];#(LU,M)
    T_RU=T_set[str(pos_RU[1-1])+','+str(pos_RU[2-1])];#(M,dRD)
    B_RU0=B_RU.clone();
    T_RU0=T_RU.clone();

    # @tensor T_RU[:]:= T_RU[-1,1,-3,-4]*O2[-5,-2,1];#M,d,R,D,virtual'
    T_RU=yastn.ncon([T_RU,O2], [[-1,1,-3,-4], [-5,-2,1]]);
    
    # U2=@ignore_derivatives unitary(fuse(space(T_RU,1)⊗space(T_RU,5)), space(T_RU,1)⊗space(T_RU,5));
    # @tensor T_RU[:]:=T_RU[1,-2,-3,-4,2]*U2[-1,1,2];##M',d,R,D
    T_RU=T_RU.fuse_legs(((0,4),1,2,3,), mode='hard');

    # O_string=@ignore_derivatives unitary(space(O1,1)',space(O1,1)');
    # @tensor B_RU[:]:=B_RU[-1,-2,-3]*O_string[-4,-5];#(L,U,M), (virtual,virtual')=>(L,U,M, virtual,virtual')
    # @tensor B_RU[:]:=B_RU[1,-2,3,2,4]*U'[1,2,-1]*U2'[3,4,-3];#L,U,M
    B_RU=yastn.ncon([B_RU,string12], [[-1,-2,-3], [-4,-5]]);
    B_RU=B_RU.fuse_legs(((0,3),1,(2,4)), mode='hard');


    # B_RU=permute(B_RU,(1,2,),(3,));
    # T_RU=permute(T_RU,(1,),(2,3,4,));
    B_RU_double= build_double_layer_swap_Tm(B_RU0.conj(),B_RU, False);#L M U
    T_RU_double= build_double_layer_swap_Bm(T_RU0.conj(),T_RU, True);#D R M
    ####################################
    # @tensor AA_LU[:]:=B_LU_double[-1,1,-4]*T_LU_double[-2,-3,1];
    # @tensor AA_RU[:]:=B_RU_double[-1,1,-4]*T_RU_double[-2,-3,1];
    AA_LU=yastn.ncon([B_LU_double,T_LU_double], [[-1,1,-4], [-2,-3,1]]);
    AA_RU=yastn.ncon([B_RU_double,T_RU_double], [[-1,1,-4], [-2,-3,1]]);
      

    AA_LU0=get_AA_simple(double_B_set,double_T_set,pos_LU);
    AA_LD0=get_AA_simple(double_B_set,double_T_set,pos_LD);
    AA_RU0=get_AA_simple(double_B_set,double_T_set,pos_RU);
    AA_RD0=get_AA_simple(double_B_set,double_T_set,pos_RD);

    ob=ob_2x2_iPESS(CTM,AA_LU,AA_RU,AA_LD0,AA_RD0,cx,cy,Lx,Ly).to_number();
    Norm=ob_2x2_iPESS(CTM,AA_LU0,AA_RU0,AA_LD0,AA_RD0,cx,cy,Lx,Ly).to_number();
    ob=ob/Norm;
    return ob


def hopping_y_iPESS_no_sign(CTM,O1,O2,string12, B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly):

    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

    #############################################

    ####
    B_RU=B_set[str(pos_RU[1-1])+','+str(pos_RU[2-1])];#(LU,M)
    T_RU=T_set[str(pos_RU[1-1])+','+str(pos_RU[2-1])];#(M,dRD)
    B_RU0=B_RU.clone();
    T_RU0=T_RU.clone();

    # @tensor T_RU[:]:= T_RU[-1,1,-3,-4]*O1[-5,-2,1];#M,d,R,D,virtual
    T_RU=yastn.ncon([T_RU,O1], [[-1,1,-3,-4], [-5,-2,1]]);
    # U1=@ignore_derivatives unitary(fuse(space(T_RU,4)⊗space(T_RU,5)), space(T_RU,4)⊗space(T_RU,5)); 
    # @tensor T_RU[:]:=T_RU[-1,-2,-3,1,2]*U1[-4,1,2];#M,d,R,D'
    T_RU=T_RU.fuse_legs((0,1,2,(3,4)), mode='hard');

    # B_RU=permute(B_RU,(1,2,),(3,));
    # T_RU=permute(T_RU,(1,),(2,3,4,));
    B_RU_double= build_double_layer_swap_Tm(B_RU0.conj(),B_RU, False);#L M U
    T_RU_double= build_double_layer_swap_Bm(T_RU0.conj(),T_RU, True);#D R M
    ####################################

    ####
    B_RD=B_set[str(pos_RD[1-1])+','+str(pos_RD[2-1])];#(LU,M)
    T_RD=T_set[str(pos_RD[1-1])+','+str(pos_RD[2-1])];#(M,dRD)
    B_RD0=B_RD.clone();
    T_RD0=T_RD.clone();

    
    # @tensor T_RD[:]:= T_RD[-1,1,-3,-4]*O2[-5,-2,1];#M,d,R,D,virtual
    T_RD=yastn.ncon([T_RD,O2], [[-1,1,-3,-4], [-5,-2,1]]);
    # U2=@ignore_derivatives unitary(fuse(space(T_RD,1)⊗space(T_RD,5)), space(T_RD,1)⊗space(T_RD,5));
    # @tensor T_RD[:]:=T_RD[1,-2,-3,-4,2]*U2[-1,1,2];#M',d,R,D
    T_RD=T_RD.fuse_legs(((0,4),1,2,3), mode='hard');

    # O_string=@ignore_derivatives unitary(space(O1,1)',space(O1,1)');
    # @tensor B_RD[:]:= B_RD[-1,-2,-3]*O_string[-4,-5];#(L,U,M), (virtual',virtual)=>(L,U,M, virtual',virtual)
    # @tensor B_RD[:]:=B_RD[-1,1,3,2,4]*U1'[1,2,-2]*U2'[3,4,-3];#L,U,M
    B_RD=yastn.ncon([B_RD,string12], [[-1,-2,-3], [-4,-5]]);
    B_RD=B_RD.fuse_legs((0,(1,3),(2,4)), mode='hard');

    # B_RD=permute(B_RD,(1,2,),(3,));
    # T_RD=permute(T_RD,(1,),(2,3,4,));

    B_RD_double= build_double_layer_swap_Tm(B_RD0.conj(),B_RD, False);#L M U
    T_RD_double= build_double_layer_swap_Bm(T_RD0.conj(),T_RD, True);#D R M
    ###################################################
    # @tensor AA_RU[:]:=B_RU_double[-1,1,-4]*T_RU_double[-2,-3,1];
    # @tensor AA_RD[:]:=B_RD_double[-1,1,-4]*T_RD_double[-2,-3,1];
    AA_RU=yastn.ncon([B_RU_double,T_RU_double], [[-1,1,-4], [-2,-3,1]]);
    AA_RD=yastn.ncon([B_RD_double,T_RD_double], [[-1,1,-4], [-2,-3,1]]);


    AA_LU0=get_AA_simple(double_B_set,double_T_set,pos_LU);
    AA_LD0=get_AA_simple(double_B_set,double_T_set,pos_LD);
    AA_RU0=get_AA_simple(double_B_set,double_T_set,pos_RU);
    AA_RD0=get_AA_simple(double_B_set,double_T_set,pos_RD);
    
    ob=ob_2x2_iPESS(CTM,AA_LU0,AA_RU,AA_LD0,AA_RD,cx,cy,Lx,Ly).to_number();
    Norm=ob_2x2_iPESS(CTM,AA_LU0,AA_RU0,AA_LD0,AA_RD0,cx,cy,Lx,Ly).to_number();
    ob=ob/Norm;
    return ob


def hopping_diagonala_iPESS_no_sign(CTM,O1,O2,string12, B_set,T_set, double_B_set, double_T_set, cx,cy,Lx,Ly):

    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

    ###################################################


    ######
    B_LD=B_set[str(pos_LD[1-1])+','+str(pos_LD[2-1])];#(LU,M)
    T_LD=T_set[str(pos_LD[1-1])+','+str(pos_LD[2-1])];#(M,dRD)
    B_LD0=B_LD.clone();
    T_LD0=T_LD.clone();

    # @tensor T_LD[:]:= T_LD[-1,1,-2,-3]*O1[-5,-4,1];#M,R,D,d,virtual
    T_LD=yastn.ncon([T_LD,O1], [[-1,1,-2,-3], [-5,-4,1]]);
    # U1=@ignore_derivatives unitary(fuse(space(T_LD,2)⊗space(T_LD,5)), space(T_LD,2)⊗space(T_LD,5)); 
    # @tensor T_LD[:]:=T_LD[-1,1,-3,-4,2]*U1[-2,1,2];#M,R',D,d
    T_LD=T_LD.fuse_legs((0,(1,4),2,3), mode='hard');
    # T_LD=permute(T_LD,(1,4,2,3,));#M,d,R',D
    T_LD=yastn.transpose(T_LD, axes=(1-1,4-1,2-1,3-1));

    # B_LD=permute(B_LD,(1,2,),(3,));
    # T_LD=permute(T_LD,(1,),(2,3,4,));

    B_LD_double= build_double_layer_swap_Tm(B_LD0.conj(),B_LD, False);#L M U
    T_LD_double= build_double_layer_swap_Bm(T_LD0.conj(),T_LD, True);#D R M

    #############################################

    ######
    B_RU=B_set[str(pos_RU[1-1])+','+str(pos_RU[2-1])];#(LU,M)
    T_RU=T_set[str(pos_RU[1-1])+','+str(pos_RU[2-1])];#(M,dRD)
    B_RU0=B_RU.clone();
    T_RU0=T_RU.clone();

    # @tensor T_RU[:]:= T_RU[-1,1,-2,-3]*O2[-5,-4,1];#M,R,D,d,virtual
    T_RU=yastn.ncon([T_RU,O2], [[-1,1,-2,-3], [-5,-4,1]]);
    # U2=@ignore_derivatives unitary(fuse(space(T_RU,3)⊗space(T_RU,5)), space(T_RU,3)⊗space(T_RU,5)); 
    # @tensor T_RU[:]:=T_RU[-1,-2,1,-4,2]*U2[-3,1,2];#M,R,D',d
    T_RU=T_RU.fuse_legs((0,1,(2,4),3), mode='hard');
    # T_RU=permute(T_RU,(1,4,2,3,));#M,d,R,D
    T_RU=yastn.transpose(T_RU, axes=(1-1,4-1,2-1,3-1));

    # B_RU=permute(B_RU,(1,2,),(3,));
    # T_RU=permute(T_RU,(1,),(2,3,4,));

    B_RU_double= build_double_layer_swap_Tm(B_RU0.conj(),B_RU, False);#L M U
    T_RU_double= build_double_layer_swap_Bm(T_RU0.conj(),T_RU, True);#D R M
    ################################################

    ######
    B_RD=B_set[str(pos_RD[1-1])+','+str(pos_RD[2-1])];#(LU,M)
    T_RD=T_set[str(pos_RD[1-1])+','+str(pos_RD[2-1])];#(M,dRD)
    B_RD0=B_RD.clone();
    T_RD0=T_RD.clone();

    # O_string=@ignore_derivatives unitary(space(O1,1),space(O1,1));
    # @tensor B_RD[:]:=B_RD[1,3,-3]*O_string[4,2]*U1'[1,2,-1]*U2'[3,4,-2];#L,U,M
    B_RD=yastn.ncon([B_RD,string12], [[-1,-2,-3], [-4,-5]]);
    B_RD=B_RD.fuse_legs(((0,3),(1,4),2), mode='hard');


    # B_RD=permute(B_RD,(1,2,),(3,));
    # T_RD=permute(T_RD,(1,),(2,3,4,));

    B_RD_double= build_double_layer_swap_Tm(B_RD0.conj(),B_RD, False);#L M U
    T_RD_double= build_double_layer_swap_Bm(T_RD0.conj(),T_RD, True);#D R M
    ################################################
    # @tensor AA_LD[:]:=B_LD_double[-1,1,-4]*T_LD_double[-2,-3,1];
    # @tensor AA_RU[:]:=B_RU_double[-1,1,-4]*T_RU_double[-2,-3,1];
    # @tensor AA_RD[:]:=B_RD_double[-1,1,-4]*T_RD_double[-2,-3,1];
    AA_LD=yastn.ncon([B_LD_double,T_LD_double], [[-1,1,-4], [-2,-3,1]]);
    AA_RU=yastn.ncon([B_RU_double,T_RU_double], [[-1,1,-4], [-2,-3,1]]);
    AA_RD=yastn.ncon([B_RD_double,T_RD_double], [[-1,1,-4], [-2,-3,1]]);
    ################################################

    AA_LU0=get_AA_simple(double_B_set,double_T_set,pos_LU);
    AA_LD0=get_AA_simple(double_B_set,double_T_set,pos_LD);
    AA_RU0=get_AA_simple(double_B_set,double_T_set,pos_RU);
    AA_RD0=get_AA_simple(double_B_set,double_T_set,pos_RD);

    ob=ob_2x2_iPESS(CTM,AA_LU0,AA_RU,AA_LD,AA_RD,cx,cy,Lx,Ly).to_number();
    Norm=ob_2x2_iPESS(CTM,AA_LU0,AA_RU0,AA_LD0,AA_RD0,cx,cy,Lx,Ly).to_number();
    ob=ob/Norm;
    return ob        



# #########################################



def evaluate_ob_cell_iPESS(parameters, B_set,T_set, double_B_set, double_T_set, CTM_cell, energy_setting, config_kwargs, global_args):
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """      
    Lx=global_args.Lx
    Ly=global_args.Ly
    with torch.no_grad(): 
        Ident, N_occu, n_double, Cdag, C, CdagC_string =Hamiltonians_spinful_Z2(config_kwargs);

    if energy_setting.model=="spinful_triangle_lattice":
    
        assert mod(Lx,2)==0
        #for 120 degree magnetic order in the Hofstadter M2 model. Unit-cell for 120 degree order should be at least 3x3. 

        t1=parameters['t1'];
        t2=parameters['t2'];
        ϕ=parameters['ϕ'];
        μ=parameters['μ'];
        U=parameters['U'];

        with torch.no_grad():
            ex_set=torch.zeros(Lx,Ly)*1j;
            ey_set=torch.zeros(Lx,Ly)*1j;
            e_diagonala_set=torch.zeros(Lx,Ly)*1j;
            e0_set=torch.zeros(Lx,Ly)*1j;
            eU_set=torch.zeros(Lx,Ly)*1j;

        
        # E_total=0;
        E_total=torch.zeros((1), dtype=B_set['1,1'].dtype,device=B_set['1,1'].device,requires_grad=True);
        for cx in range(1,Lx+1):
            for cy in range(1,Ly+1):

                ex=checkpoint(hopping_x_iPESS, CTM_cell, Cdag, C, CdagC_string, B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly, use_reentrant=False);
                ey=checkpoint(hopping_y_iPESS, CTM_cell, Cdag, C, CdagC_string, B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly, use_reentrant=False);
                e_diagonala=checkpoint(hopping_diagonala_iPESS, CTM_cell, Cdag, C, CdagC_string, B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly, use_reentrant=False);
                e0=checkpoint(ob_onsite_iPESS, CTM_cell,N_occu, B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly, use_reentrant=False);
                eU=checkpoint(ob_onsite_iPESS, CTM_cell,n_double-(1/2)*N_occu+(1/4)*Ident,B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly, use_reentrant=False);
                with torch.no_grad():
                    ex_set[cx-1,cy-1]=ex;
                    ey_set[cx-1,cy-1]=ey;
                    e_diagonala_set[cx-1,cy-1]=e_diagonala;
                    e0_set[cx-1,cy-1]=e0;
                    eU_set[cx-1,cy-1]=eU;
                if mod(cx,2)==1:
                    E_total=E_total+torch.real(t1*(cmath.exp(1j*ϕ)*ex)*2-t1*(ey)*2-t2*(e_diagonala)*2 -μ*e0 +U*eU);
                else:
                    E_total=E_total+torch.real(t1*(cmath.exp(1j*ϕ)*ex)*2+t1*(ey)*2+t2*(e_diagonala)*2 -μ*e0 +U*eU);


        E_Bz=(torch.zeros(1,1)*1j).to(E_total.device);
        if 'Bz' in parameters:
            if abs(parameters['Bz'])>0:
                Bz=parameters['Bz'];
                sx_op,sy_op,sz_op=spin_operator_Z2(config_kwargs);
                for cx in range(1,Lx+1):
                    for cy in range(1,Ly+1):
                        e_sz=checkpoint(ob_onsite_iPESS, CTM_cell, sz_op, B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly, use_reentrant=False);
                        
                        E_total=E_total+Bz*e_sz;
                        E_Bz=E_Bz+Bz*e_sz;
        # print('Bz energy:')
        # print(E_Bz)
        E_total=E_total/(Lx*Ly);
        return torch.real(E_total),  ex_set, ey_set, e_diagonala_set, e0_set, eU_set
    
    elif energy_setting.model =="standard_triangle_Hubbard":    
        t1=parameters['t1'];
        t2=parameters['t2'];
        μ=parameters['μ'];
        U=parameters['U'];

        with torch.no_grad():
            ex_set=torch.zeros(Lx,Ly)*1j;
            ey_set=torch.zeros(Lx,Ly)*1j;
            e_diagonala_set=torch.zeros(Lx,Ly)*1j;
            e0_set=torch.zeros(Lx,Ly)*1j;
            eU_set=torch.zeros(Lx,Ly)*1j;
        
        # E_total=0;
        E_total=torch.zeros((1), dtype=B_set['1,1'].dtype,device=B_set['1,1'].device,requires_grad=True);
        for cx in range(1,Lx+1):
            for cy in range(1,Ly+1):
                #(cx,cy): coordinate of left-top C1 tensor
                ex=checkpoint(hopping_x_iPESS,CTM_cell, Cdag, C, CdagC_string, B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly, use_reentrant=False);
                ey=checkpoint(hopping_y_iPESS,CTM_cell, Cdag, C, CdagC_string, B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly, use_reentrant=False);
                e_diagonala=checkpoint(hopping_diagonala_iPESS,CTM_cell, Cdag, C, CdagC_string, B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly, use_reentrant=False);
                e0=checkpoint(ob_onsite_iPESS,CTM_cell,N_occu, B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly, use_reentrant=False);
                eU=checkpoint(ob_onsite_iPESS,CTM_cell,n_double-(1/2)*N_occu+(1/4)*Ident, B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly, use_reentrant=False);
                with torch.no_grad():
                    ex_set[cx-1,cy-1]=ex;
                    ey_set[cx-1,cy-1]=ey;
                    e_diagonala_set[cx-1,cy-1]=e_diagonala;
                    e0_set[cx-1,cy-1]=e0;
                    eU_set[cx-1,cy-1]=eU;

                E_temp=-t1*ex -t1*ey -t2*e_diagonala -μ*e0/2  +U*eU/2;
                #E_temp=-t1*ex -t1*ey -t2*e_diagonala  +U*eU/2; # do not include chemical potential
                E_total=E_total+torch.real(E_temp)*2;
                
        E_Bz=(torch.zeros(1,1)*1j).to(E_total.device);
        if 'Bz' in parameters:
            if abs(parameters['Bz'])>0:
                Bz=parameters['Bz'];
                sx_op,sy_op,sz_op=spin_operator_Z2(config_kwargs);
                for cx in range(1,Lx+1):
                    for cy in range(1,Ly+1):
                        e_sz=checkpoint(ob_onsite_iPESS, CTM_cell, sz_op, B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly, use_reentrant=False);
                        
                        E_total=E_total+Bz*e_sz;
                        E_Bz=E_Bz+Bz*e_sz;
        # print('Bz energy:')
        # print(E_Bz)

        E_total=E_total/(Lx*Ly);
        return torch.real(E_total),  ex_set, ey_set, e_diagonala_set, e0_set, eU_set
    


def evaluate_spin_cell_iPESS(B_set,T_set, double_B_set, double_T_set, CTM_cell, config_kwargs, global_args):
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """    
    Lx=global_args.Lx;
    Ly=global_args.Ly;
    with torch.no_grad():
        sx_op,sy_op,sz_op=spin_operator_Z2(config_kwargs);

        sx_set=torch.zeros(Lx,Ly)*1j;
        sy_set=torch.zeros(Lx,Ly)*1j;
        sz_set=torch.zeros(Lx,Ly)*1j;
        for cx in range(1,Lx+1):
            for cy in range(1,Ly+1):
                #(cx,cy): coordinate of left-top C1 tensor

                sx0=ob_onsite_iPESS(CTM_cell,sx_op,B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly);
                sy0=ob_onsite_iPESS(CTM_cell,sy_op,B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly);
                sz0=ob_onsite_iPESS(CTM_cell,sz_op,B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly);
                sx_set[cx-1,cy-1]=sx0;
                sy_set[cx-1,cy-1]=sy0;
                sz_set[cx-1,cy-1]=sz0;

        return sx_set,sy_set,sz_set






def evaluate_spin_ob_cell_iPESS(B_set,T_set, double_B_set, double_T_set, CTM_cell, energy_setting, config_kwargs, global_args):
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """    
    Lx=global_args.Lx
    Ly=global_args.Ly
    with torch.no_grad():
        Sa, Sb, SS_string, chirality_S1, chirality_S2, chirality_S3, chirality_string12, chirality_string23 =Operators_spinful_Z2(config_kwargs);
 
        triangle_up_set=torch.zeros(Lx,Ly)*1j;
        triangle_dn_set=torch.zeros(Lx,Ly)*1j;

        SS_x_set=torch.zeros(Lx,Ly)*1j;
        SS_y_set=torch.zeros(Lx,Ly)*1j;
        SS_diagonal_set=torch.zeros(Lx,Ly)*1j;


        for cx in range(1,Lx+1):
            for cy in range(1,Ly+1):

                #expectation value for chirality operator
                up_triangle=ob_up_triangle_iPESS(CTM_cell,chirality_S1,chirality_S2,chirality_S3, chirality_string12, chirality_string23, B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly);#LD,RD,RU
                dn_triangle=-ob_dn_triangle_iPESS(CTM_cell,chirality_S1,chirality_S2,chirality_S3, chirality_string12, chirality_string23,B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly);#LD,LU,RU

                #expectation value for Heisenberg operator
                SS_x=hopping_x_iPESS_no_sign(CTM_cell,Sa,Sb,SS_string,B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly);
                SS_y=hopping_y_iPESS_no_sign(CTM_cell,Sa,Sb,SS_string,B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly);
                SS_diagonal=hopping_diagonala_iPESS_no_sign(CTM_cell,Sa,Sb,SS_string,B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly);

                triangle_up_set[cx-1,cy-1]=up_triangle;
                triangle_dn_set[cx-1,cy-1]=dn_triangle;
                SS_x_set[cx-1,cy-1]=SS_x;
                SS_y_set[cx-1,cy-1]=SS_y;
                SS_diagonal_set[cx-1,cy-1]=SS_diagonal;


        return triangle_up_set,triangle_dn_set,SS_x_set,SS_y_set,SS_diagonal_set
    


def evaluate_ob_pairing_cell(B_set,T_set, double_B_set, double_T_set, CTM_cell, config_kwargs, global_args):
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """    
    Lx=global_args.Lx
    Ly=global_args.Ly
    with torch.no_grad():
        pairing_singlet_site1, pairing_singlet_site2, pairing_string=twosite_pairing_spinful_Z2(config_kwargs);

        pairing_x_set=torch.zeros(Lx,Ly)*1j;
        pairing_y_set=torch.zeros(Lx,Ly)*1j;
        pairing_diagonal_set=torch.zeros(Lx,Ly)*1j;


        for cx in range(1,Lx+1):
            for cy in range(1,Ly+1):

                #expectation value for Heisenberg operator
                pairing_x=hopping_x_iPESS(CTM_cell, pairing_singlet_site1, pairing_singlet_site2, pairing_string, B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly);
                pairing_y=hopping_y_iPESS(CTM_cell, pairing_singlet_site1, pairing_singlet_site2, pairing_string, B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly);
                pairing_diagonal=hopping_diagonala_iPESS(CTM_cell, pairing_singlet_site1, pairing_singlet_site2, pairing_string, B_set,T_set, double_B_set, double_T_set,cx,cy,Lx,Ly);

                pairing_x_set[cx-1,cy-1]=pairing_x;
                pairing_y_set[cx-1,cy-1]=pairing_y;
                pairing_diagonal_set[cx-1,cy-1]=pairing_diagonal;


        return pairing_x_set,pairing_y_set,pairing_diagonal_set
    

