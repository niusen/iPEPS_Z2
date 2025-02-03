import yastn
import numpy,torch
import sys
import copy
from collections import OrderedDict
from config.settings import *

def spectrum_conv_check(ss_old,C_new):
    U,spec,V=yastn.linalg.svd(C_new,  svd_on_cpu=True)
    spec=spec.to_dense();
    ss_new=torch.diag(spec/spec[0,0]);
    ss_new=ss_new.to(ss_old.device);

    if len(ss_old)>len(ss_new):
        dss=copy.deepcopy(ss_old);
        siz=len(ss_new)
    elif len(ss_old)<=len(ss_new):
        dss=copy(ss_new);
        siz=len(ss_old)
    
    # print(ss_old.device)
    # print(ss_new.device)
    dss[0:siz]=ss_old[0:siz]-ss_new[0:siz]
    er=torch.Tensor.norm(dss);
    return er,ss_new


def build_double_layer_swap_Tm(Ap,A, with_physical):
    if ~with_physical: #no physical leg
        # assert(len(A.s)==3)#LU,M
        # assert(len(Ap.s)==3)
        assert(Ap.get_rank()==3);
        assert(A.get_rank()==3);

        # gate=@ignore_derivatives swap_gate(Ap,2,3); #gate L'U'
        # @tensor Ap[:]:=Ap[-1,1,2]*gate[-2,-3,1,2];  
        Ap=yastn.swap_gate(Ap, (1-1,2-1));

        # gate=@ignore_derivatives parity_gate(Ap,1); #gate M'
        # @tensor Ap[:]:=Ap[1,-2,-3]*gate[-1,1];
        # gate=@ignore_derivatives parity_gate(Ap,3); #gate U'
        # @tensor Ap[:]:=Ap[-1,-2,1]*gate[-3,1];
        Ap=yastn.swap_gate(Ap, (3-1,3-1));
        Ap=yastn.swap_gate(Ap, (2-1,2-1));

        
        # A=permute(A,(1,2,),(3,));
        # Ap=permute(Ap,(1,),(2,3,));
        

        # U_L=@ignore_derivatives unitary(fuse(space(Ap, 2) ⊗ space(A, 1)), space(Ap, 2) ⊗ space(A, 1));
        # U_D=@ignore_derivatives unitary(fuse(space(Ap, 1) ⊗ space(A, 3)), space(Ap, 1) ⊗ space(A, 3));
        # U_U=@ignore_derivatives unitary(space(Ap, 3)' ⊗ space(A, 2)', fuse(space(Ap, 3)' ⊗ space(A, 2)'));

        # @tensor AA_fused[:]:=Ap[5,1,3]*A[2,4,6]*U_L[-1,1,2]*U_D[-2,5,6]*U_U[3,4,-3];
        AA_fused = yastn.ncon([Ap, A], [[-1, -3,-5], [-2, -4,-6]]);#L'L,U',U,M',M
        
    

        # P_odd_Lp,_=@ignore_derivatives projector_parity(space(U_L',1));
        # P_odd_Up,_=@ignore_derivatives projector_parity(space(U_U',2));
        # P_odd_U,_=@ignore_derivatives projector_parity(space(U_U',3));

        # @tensor isom_Lp[:]:=U_L[-1,4,3]*P_odd_Lp'[4,1]*P_odd_Lp[1,2]*U_L'[2,3,-2];
        # @tensor isom_U[:]:=U_U[3,4,-1]*P_odd_U'[4,1]*P_odd_U[1,2]*U_U'[-2,3,2];
        # @tensor isom_Up_U[:]:=U_U[3,4,-1]*P_odd_Up'[3,1]*P_odd_Up[1,5]*P_odd_U'[4,2]*P_odd_U[2,6]*U_U'[-2,5,6];
        # @tensor AA_Lp_U[:]:=AA_fused[1,-2,4]*isom_Lp[-1,1]*isom_U[-3,4];
        # AA_fused=AA_fused-2*AA_Lp_U;
        # @tensor AA_Up_U[:]:=AA_fused[-1,-2,4]*isom_Up_U[-3,4];
        # AA_fused=AA_fused-2*AA_Up_U;

        AA_fused=yastn.swap_gate(AA_fused, (1-1,4-1));#L',U
        AA_fused=yastn.swap_gate(AA_fused, (3-1,4-1));#U',U


        # P_odd_Dp,_=@ignore_derivatives projector_parity(space(U_D',1));
        # P_odd_D,_=@ignore_derivatives projector_parity(space(U_D',2));
        # @tensor isom_Dp[:]:=U_D[-1,4,3]*P_odd_Dp'[4,1]*P_odd_Dp[1,2]*U_D'[2,3,-2];
        # @tensor isom_Dp_D[:]:=U_D[-1,3,4]*P_odd_Dp'[3,1]*P_odd_Dp[1,5]*P_odd_D'[4,2]*P_odd_D[2,6]*U_D'[5,6,-2];
        # @tensor AA_Dp_D[:]:=AA_fused[-1,2,-3]*isom_Dp_D[-2,2];
        # AA_fused=AA_fused-2*AA_Dp_D;

        AA_fused=yastn.swap_gate(AA_fused, (5-1,6-1));#M',M
  

        AA_fused=AA_fused.fuse_legs(((0,1),(4,5),(2,3,)), mode='hard');#L,M,U


        #double layer order: L M U = L D U 
        return AA_fused


def build_double_layer_swap_Bm(Ap,A, with_physical):
    if with_physical: #with one physical leg
        # assert (length(codomain(A))==1)&(length(domain(A))==3)
        # assert (length(codomain(Ap))==3)&(length(domain(Ap))==1)
        assert(Ap.get_rank()==4);
        assert(A.get_rank()==4);
        #treat (M,dRD) as (U,dRD)
        #treat (d'R'D',M') as (d'R'D',U')
        # println(space(Ap))
        # println(space(A))


        # gate=@ignore_derivatives swap_gate(Ap,2,3); #gate D'R'
        # @tensor Ap[:]:=Ap[-1,1,2,-4]*gate[-2,-3,1,2];  
        # gate=@ignore_derivatives parity_gate(Ap,4); #gate M'
        # @tensor Ap[:]:=Ap[-1,-2,-3,1]*gate[-4,1];
        # gate=@ignore_derivatives parity_gate(Ap,3);  #gate D'
        # @tensor Ap[:]:=Ap[-1,-2,1,-4]*gate[-3,1];

        Ap=yastn.swap_gate(Ap, (3-1,4-1));#D'R'
        Ap=yastn.swap_gate(Ap, (1-1,1-1));#M'
        Ap=yastn.swap_gate(Ap, (4-1,4-1));#D'
        

        
    
        # U_D=@ignore_derivatives unitary(fuse(space(Ap, 3) ⊗ space(A, 4)), space(Ap, 3) ⊗ space(A, 4));
        # U_R=@ignore_derivatives unitary(space(Ap, 2)' ⊗ space(A, 3)', fuse(space(Ap, 2)' ⊗ space(A, 3)'));
        # U_U=@ignore_derivatives unitary(space(Ap, 4)' ⊗ space(A, 1)', fuse(space(Ap, 4)' ⊗ space(A, 1)'));


        # @tensor AA_fused[:]:=Ap[3,1,6,4]*A[5,3,2,7]*U_U[4,5,-3]*U_R[1,2,-2]*U_D[-1,6,7];
        AA_fused = yastn.ncon([Ap, A], [[-1,1, -3,-5], [-2,1, -4,-6]]);#M',M,R',R,D',D
        


        # P_odd_Up,_=@ignore_derivatives projector_parity(space(U_U',2));
        # P_odd_U,_=@ignore_derivatives projector_parity(space(U_U',3));
        # @tensor isom_U[:]:=U_U[3,4,-1]*P_odd_U'[4,1]*P_odd_U[1,2]*U_U'[-2,3,2];
        # @tensor isom_Up_U[:]:=U_U[3,4,-1]*P_odd_Up'[3,1]*P_odd_Up[1,5]*P_odd_U'[4,2]*P_odd_U[2,6]*U_U'[-2,5,6];
        # @tensor AA_Up_U[:]:=AA_fused[-1,-2,4]*isom_Up_U[-3,4];
        # AA_fused=AA_fused-2*AA_Up_U;
        AA_fused=yastn.swap_gate(AA_fused, (1-1,2-1));#U',U



        # P_odd_Dp,_=@ignore_derivatives projector_parity(space(U_D',1));
        # P_odd_D,_=@ignore_derivatives projector_parity(space(U_D',2));
        # P_odd_R,_=@ignore_derivatives projector_parity(space(U_R',3));
        # @tensor isom_Dp[:]:=U_D[-1,4,3]*P_odd_Dp'[4,1]*P_odd_Dp[1,2]*U_D'[2,3,-2];
        # @tensor isom_R[:]:=U_R[3,4,-1]*P_odd_R'[4,1]*P_odd_R[1,2]*U_R'[-2,3,2];
        # @tensor isom_Dp_D[:]:=U_D[-1,3,4]*P_odd_Dp'[3,1]*P_odd_Dp[1,5]*P_odd_D'[4,2]*P_odd_D[2,6]*U_D'[5,6,-2];
        # @tensor AA_Dp_D[:]:=AA_fused[2,-2,-3]*isom_Dp_D[-1,2];
        # AA_fused=AA_fused-2*AA_Dp_D;
        # @tensor AA_Dp_R[:]:=AA_fused[2,3,-3]*isom_Dp[-1,2]*isom_R[-2,3];
        # AA_fused=AA_fused-2*AA_Dp_R;
        AA_fused=yastn.swap_gate(AA_fused, (5-1,6-1));#D',D
        AA_fused=yastn.swap_gate(AA_fused, (4-1,5-1));#D',R

        AA_fused=AA_fused.fuse_legs(((4,5),(2,3),(0,1,)), mode='hard');#D,R,M
        #double layer order: D R M = D R U 
        return AA_fused



    ##########################


def build_doublelayer_swap_iPESS(B_set,T_set, pos):
    c1=pos[1-1];
    c2=pos[2-1];
    B=B_set[str(c1)+','+str(c2)];
    B_double= build_double_layer_swap_Tm(B.conj(),B, False);#L M U
    T=T_set[str(c1)+','+str(c2)];
    T_double = build_double_layer_swap_Bm(T.conj(),T, True);#D R M
    # @tensor AA[:]:=B_double[-1,1,-4]*T_double[-2,-3,1];#(L M U),(D R M) =>(L,D,R,U)
    # AA = yastn.ncon([B_double, T_double], [[-1,1, -4], [-2,-3,1]]);
    return T_double, B_double



def convert_cell_posit(cx,cy,dx,dy,direction,Lx,Ly):

    if direction==1:
        posit=(mod1(cx+dx,Lx),mod1(cy+dy,Ly));
    elif direction==2:
        posit=(mod1(cy-dy,Lx),mod1(cx+dx,Ly));
    elif direction==3:
        posit=(mod1(cx-dx,Lx),mod1(cy-dy,Ly));
    elif direction==4:
        posit=(mod1(cy+dy,Lx),mod1(cx-dx,Ly));
    return posit


def rotate_AA_direction(AA_fused,direction):
    #AA_rotated=permute(AA_fused, (mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),),());
    AA_rotated=yastn.transpose(AA_fused, axes=(mod1(2-direction,4)-1,mod1(3-direction,4)-1,mod1(4-direction,4)-1,mod1(1-direction,4)-1))
    return AA_rotated




def Fermionic_CTMRG_cell_iPESS(B_set,T_set,init,CTM0, ctm_setting,global_args):
    chi=ctm_setting.chi;
    #Ref: PHYSICAL REVIEW B 98, 235148 (2018)
    ########################
    CTM_ite_info=ctm_setting.CTM_ite_info;
    projector_strategy=ctm_setting.projector_strategy;
    CTM_trun_svd=ctm_setting.CTM_trun_svd;
    CTM_ite_nums=ctm_setting.CTM_ite_nums;
    #######################
    Lx=global_args.Lx;
    Ly=global_args.Ly;
    #######################

    if (CTM_trun_svd==True) & (projector_strategy=="4x4"):
        print("Attention: truncated svd with 4x4 projector could give large error");
    

    #initial corner transfer matrix
    if init.reconstruct_AA:
        double_B_cell=initial_cell(Lx,Ly);
        double_T_cell=initial_cell(Lx,Ly);
        # U_L_cell=initial_tuple_cell(Lx,Ly);
        # U_D_cell=initial_tuple_cell(Lx,Ly);
        # U_R_cell=initial_tuple_cell(Lx,Ly);
        # U_U_cell=initial_tuple_cell(Lx,Ly);
        for cx in range(1,Lx+1):
            for cy in range(1,Ly+1):
                T_double, B_double =build_doublelayer_swap_iPESS(B_set,T_set, (cx,cy));

                double_B_cell[str(cx)+','+str(cy)]= B_double;
                double_T_cell[str(cx)+','+str(cy)]= T_double;
                # U_L_cell=fill_tuple(U_L_cell, U_L_, cx,cy);
                # U_D_cell=fill_tuple(U_D_cell, U_D_, cx,cy);
                # U_R_cell=fill_tuple(U_R_cell, U_R_, cx,cy);
                # U_U_cell=fill_tuple(U_U_cell, U_U_, cx,cy);
        #     end
        # end
        # AA_memory= (Base.summarysize(double_B_cell)+Base.summarysize(double_T_cell))/1024/1024;
        # if CTM_ite_info:
        #     print("Memory cost of double layer iPESS: "*string(AA_memory)*" Mb.");flush(stdout);

    # else
        # AA_fused_cell=auxi_tensors.AA_fused_cell;
        # U_L_cell=auxi_tensors.U_L_cell;
        # U_D_cell=auxi_tensors.U_D_cell;
        # U_R_cell=auxi_tensors.U_R_cell;
        # U_U_cell=auxi_tensors.U_U_cell;
    # end

    if init.reconstruct_CTM:
        CTM_cell= init_CTM_cell(B_set,T_set,ctm_setting,global_args);
    else:
        #copy.deepcopy is not for autograd
        CTM_cell=copy.deepcopy(CTM0);
    # end
    
    ss_old1_cell= torch.ones((Lx,Ly,chi*2),dtype=torch.float64, device=B_set['1,1'].device);
    ss_old2_cell= torch.ones((Lx,Ly,chi*2),dtype=torch.float64, device=B_set['1,1'].device);
    ss_old3_cell= torch.ones((Lx,Ly,chi*2),dtype=torch.float64, device=B_set['1,1'].device);
    ss_old4_cell= torch.ones((Lx,Ly,chi*2),dtype=torch.float64, device=B_set['1,1'].device);
    ss_new1_cell= torch.zeros((Lx,Ly,chi*2),dtype=torch.float64, device=B_set['1,1'].device);
    ss_new2_cell= torch.zeros((Lx,Ly,chi*2),dtype=torch.float64, device=B_set['1,1'].device);
    ss_new3_cell= torch.zeros((Lx,Ly,chi*2),dtype=torch.float64, device=B_set['1,1'].device);
    ss_new4_cell= torch.zeros((Lx,Ly,chi*2),dtype=torch.float64, device=B_set['1,1'].device);
    er1_cell= torch.zeros((Lx,Ly));
    er2_cell= torch.zeros((Lx,Ly));
    er3_cell= torch.zeros((Lx,Ly));
    er4_cell= torch.ones((Lx,Ly));


    Cset_cell=CTM_cell['Cset'];
    Tset_cell=CTM_cell['Tset'];
    conv_check="singular_value"



    #     end
    # end


    #Iteration
    with torch.no_grad():
        print_corner=False;
        C1_spec_cell=torch.zeros((Lx,Ly,chi*2)).to(dtype=ss_old1_cell.dtype,device=ss_old1_cell.device);
        C2_spec_cell=torch.zeros((Lx,Ly,chi*2)).to(dtype=ss_old1_cell.dtype,device=ss_old1_cell.device);
        C3_spec_cell=torch.zeros((Lx,Ly,chi*2)).to(dtype=ss_old1_cell.dtype,device=ss_old1_cell.device);
        C4_spec_cell=torch.zeros((Lx,Ly,chi*2)).to(dtype=ss_old1_cell.dtype,device=ss_old1_cell.device);
        if print_corner:
            for cx in range(1,Lx+1):
                for cy in range(1,Ly+1):
                    
                    print("cell position: "+str([cx,cy]))
                    print("corner 4:")
                    uu,C4_spec,vv=yastn.linalg.svd(Cset_cell[str(cx)+','+str(cy)]['C4'], axes=(0, 1), svd_on_cpu=True);
                    C4_spec=C4_spec.to_dense();
                    C4_spec=torch.diag(C4_spec/C4_spec[0,0]);
                    C4_spec_cell[cx-1,cy-1,:]=C4_spec_cell[cx-1,cy-1,:]*0;
                    C4_spec_cell[cx-1,cy-1,range(0,len(C4_spec))]=C4_spec;
                    print("corner 1:")
                    uu,C1_spec,vv=yastn.linalg.svd(Cset_cell[str(cx)+','+str(cy)]['C1'], axes=(0, 1), svd_on_cpu=True);
                    C1_spec=C1_spec.to_dense();
                    C1_spec=torch.diag(C1_spec/C1_spec[0,0]);
                    C1_spec_cell[cx-1,cy-1,:]=C1_spec_cell[cx-1,cy-1,:]*0;
                    C1_spec_cell[cx-1,cy-1,range(0,len(C1_spec))]=C1_spec;
                    print(C1_spec);
                    print("corner 3:")
                    uu,C3_spec,vv=yastn.linalg.svd(Cset_cell[str(cx)+','+str(cy)]['C3'], axes=(0, 1), svd_on_cpu=True);
                    C3_spec=C3_spec.to_dense();
                    C3_spec=torch.diag(C3_spec/C3_spec[0,0]);
                    C3_spec_cell[cx-1,cy-1,:]=C3_spec_cell[cx-1,cy-1,:]*0;
                    C3_spec_cell[cx-1,cy-1,range(0,len(C3_spec))]=C3_spec;
                    print(C3_spec);
                    print("corner 2:")
                    uu,C2_spec,vv=yastn.linalg.svd(Cset_cell[str(cx)+','+str(cy)]['C2'], axes=(0, 1), svd_on_cpu=True);
                    C2_spec=C2_spec.to_dense();
                    C2_spec=torch.diag(C2_spec/C2_spec[0,0]);
                    C2_spec_cell[cx-1,cy-1,:]=C2_spec_cell[cx-1,cy-1,:]*0;
                    C2_spec_cell[cx-1,cy-1,range(0,len(C2_spec))]=C2_spec;
                    print(C2_spec);
    
            print("CTM init finished")

    


    if CTM_ite_info:
        print("start CTM iterations:")
    # end
    ite_num=0;
    ite_err=1;
    err_set=(1,);#tuple
    

    CTM_ite_cell=CTM_ite_cell_continuous_update;
    

    for ci in range(1,CTM_ite_nums+1):
        ite_num=ci;
        #direction_order=[1,2,3,4];
        #direction_order=[4,1,2,3];
        direction_order=[3,4,1,2];
        
        for direction in direction_order:
            Cset_cell,Tset_cell=CTM_ite_cell(Cset_cell, Tset_cell, double_B_cell,double_T_cell, chi, direction,ctm_setting,global_args);
        # end
        
        with torch.no_grad():
            print_corner=False;
            if print_corner:
                for cx in range(1,Lx+1):
                    for cy in range(1,Ly+1):
                        
                        print("cell position: "+str([cx,cy]))
                        print("corner 4:")
                        uu,C4_spec,vv=yastn.linalg.svd(Cset_cell[str(cx)+','+str(cy)]['C4'], svd_on_cpu=True);
                        C4_spec=C4_spec.to_dense();
                        C4_spec=torch.diag(C4_spec/C4_spec[0,0]);
                        C4_spec_cell[cx-1,cy-1,:]=C4_spec_cell[cx-1,cy-1,:]*0;
                        C4_spec_cell[cx-1,cy-1,range(0,len(C4_spec))]=C4_spec;
                        print(C4_spec);
                        print("corner 1:")
                        uu,C1_spec,vv=yastn.linalg.svd(Cset_cell[str(cx)+','+str(cy)]['C1'], svd_on_cpu=True);
                        C1_spec=C1_spec.to_dense();
                        C1_spec=torch.diag(C1_spec/C1_spec[0,0]);
                        C1_spec_cell[cx-1,cy-1,:]=C1_spec_cell[cx-1,cy-1,:]*0;
                        C1_spec_cell[cx-1,cy-1,range(0,len(C1_spec))]=C1_spec;
                        print(C1_spec);
                        print("corner 3:")
                        uu,C3_spec,vv=yastn.linalg.svd(Cset_cell[str(cx)+','+str(cy)]['C3'], svd_on_cpu=True);
                        C3_spec=C3_spec.to_dense();
                        C3_spec=torch.diag(C3_spec/C3_spec[0,0]);
                        C3_spec_cell[cx-1,cy-1,:]=C3_spec_cell[cx-1,cy-1,:]*0;
                        C3_spec_cell[cx-1,cy-1,range(0,len(C3_spec))]=C3_spec;
                        print(C3_spec);
                        print("corner 2:")
                        uu,C2_spec,vv=yastn.linalg.svd(Cset_cell[str(cx)+','+str(cy)]['C2'], svd_on_cpu=True);
                        C2_spec=C2_spec.to_dense();
                        C2_spec=torch.diag(C2_spec/C2_spec[0,0]);
                        C2_spec_cell[cx-1,cy-1,:]=C2_spec_cell[cx-1,cy-1,:]*0;
                        C2_spec_cell[cx-1,cy-1,range(0,len(C2_spec))]=C2_spec;
                        print(C2_spec);
                print("next iteration:")
        
        

        with torch.no_grad():
            if conv_check=="singular_value": #check convergence of singular value
                for cx in range(1,Lx+1):
                    for cy in range(1,Ly+1):
                        
                        er1,ss_new1=spectrum_conv_check(ss_old1_cell[cx-1,cy-1,:],Cset_cell[str(cx)+','+str(cy)]['C1']);
                        er2,ss_new2=spectrum_conv_check(ss_old2_cell[cx-1,cy-1,:],Cset_cell[str(cx)+','+str(cy)]['C2']);
                        er3,ss_new3=spectrum_conv_check(ss_old3_cell[cx-1,cy-1,:],Cset_cell[str(cx)+','+str(cy)]['C3']);
                        er4,ss_new4=spectrum_conv_check(ss_old4_cell[cx-1,cy-1,:],Cset_cell[str(cx)+','+str(cy)]['C4']);

                        # println([cx,cy])
                        # println(ss_new1)
                        # println(ss_new2)
                        # println(ss_new3)
                        # println(ss_new4)
                        

                        er1_cell[cx-1,cy-1]=er1;
                        er2_cell[cx-1,cy-1]=er2;
                        er3_cell[cx-1,cy-1]=er3;
                        er4_cell[cx-1,cy-1]=er4;
                        ss_new1_cell[cx-1,cy-1,:]=ss_new1_cell[cx-1,cy-1,:]*0;
                        ss_new2_cell[cx-1,cy-1,:]=ss_new2_cell[cx-1,cy-1,:]*0;
                        ss_new3_cell[cx-1,cy-1,:]=ss_new3_cell[cx-1,cy-1,:]*0;
                        ss_new4_cell[cx-1,cy-1,:]=ss_new4_cell[cx-1,cy-1,:]*0;
                        ss_new1_cell[cx-1,cy-1,range(0,len(ss_new1))]=ss_new1;
                        ss_new2_cell[cx-1,cy-1,range(0,len(ss_new2))]=ss_new2;
                        ss_new3_cell[cx-1,cy-1,range(0,len(ss_new3))]=ss_new3;
                        ss_new4_cell[cx-1,cy-1,range(0,len(ss_new4))]=ss_new4;

            #     end
            # end

            with torch.no_grad():
                er=max((torch.max(er1_cell.flatten()),torch.max(er2_cell.flatten()),torch.max(er3_cell.flatten()),torch.max(er4_cell.flatten())));
                er=er.item();
                err_set=err_set+(er,);#connect tuple

                ite_err=er;
                if CTM_ite_info:
                    print("CTMRG iteration: "+str(ci)+", CTMRG err: "+str(er));
                # end
                if er<ctm_setting.CTM_conv_tol:
                    break;
                # end


                ss_old1_cell=ss_new1_cell;
                ss_old2_cell=ss_new2_cell;
                ss_old3_cell=ss_new3_cell;
                ss_old4_cell=ss_new4_cell;
        

    # CTM_cell={'Cset':Cset_cell,'Tset':Tset_cell};
    CTM_cell=OrderedDict()
    CTM_cell['Cset']=Cset_cell;
    CTM_cell['Tset']=Tset_cell;
    return CTM_cell, double_B_cell,double_T_cell,ite_num,ite_err

def get_AA_direction(double_B_cell,double_T_cell,direction,pos):
    B_double=double_B_cell[str(pos[1-1])+','+str(pos[2-1])];
    T_double=double_T_cell[str(pos[1-1])+','+str(pos[2-1])];
    # @tensor AA[:]:=B_double[-1,1,-4]*T_double[-2,-3,1];
    AA = yastn.ncon([B_double, T_double], [[-1,1,-4], [-2,-3,1]])
    return rotate_AA_direction(AA,direction)



def build_corner_MMup(coord_,direction_,double_B_cell_,double_T_cell_,Cset_cell_,Tset_cell_,Lx,Ly):
    
    Pos=convert_cell_posit(coord_[1-1],coord_[2-1],1,1,direction_,Lx,Ly);
    AA=get_AA_direction(double_B_cell_,double_T_cell_,direction_,Pos);
    Pos=convert_cell_posit(coord_[1-1],coord_[2-1],0,0,direction_,Lx,Ly);
    C1=Cset_cell_[str(Pos[1-1])+','+str(Pos[2-1])]['C'+str(mod1(direction_,4))];
    Pos=convert_cell_posit(coord_[1-1],coord_[2-1],1,0,direction_,Lx,Ly);
    T1=Tset_cell_[str(Pos[1-1])+','+str(Pos[2-1])]['T'+str(mod1(direction_,4))];
    Pos=convert_cell_posit(coord_[1-1],coord_[2-1],0,1,direction_,Lx,Ly);
    T4=Tset_cell_[str(Pos[1-1])+','+str(Pos[2-1])]['T'+str(mod1(direction_-1,4))];
    #@tensor MMup_[:]:=C1[1,2]*T1[2,3,-3]*T4[-1,4,1]*AA[4,-2,-4,3];
    MMup_ = yastn.ncon([C1, T1, T4, AA], [[1,2,], [2,3,-3],[-1,4,1], [4,-2,-4,3]])
    return MMup_


def build_corner_MMlow(coord_,direction_,double_B_cell_,double_T_cell_,Cset_cell_,Tset_cell_,Lx,Ly):

    Pos=convert_cell_posit(coord_[1-1],coord_[2-1],1,2,direction_,Lx,Ly);
    AA=get_AA_direction(double_B_cell_,double_T_cell_,direction_,Pos);
    Pos=convert_cell_posit(coord_[1-1],coord_[2-1],0,2,direction_,Lx,Ly);
    T4=Tset_cell_[str(Pos[1-1])+','+str(Pos[2-1])]['T'+str(mod1(direction_-1,4))];
    Pos=convert_cell_posit(coord_[1-1],coord_[2-1],0,3,direction_,Lx,Ly);
    C4=Cset_cell_[str(Pos[1-1])+','+str(Pos[2-1])]['C'+str(mod1(direction_-1,4))];
    Pos=convert_cell_posit(coord_[1-1],coord_[2-1],1,3,direction_,Lx,Ly);
    T3=Tset_cell_[str(Pos[1-1])+','+str(Pos[2-1])]['T'+str(mod1(direction_-2,4))];
    # @tensor MMlow_[:]:=T4[1,3,-1]*AA[3,4,-4,-2]*C4[2,1]*T3[-3,4,2];
    MMlow_ = yastn.ncon([T4, AA, C4, T3], [[1,3,-1], [3,4,-4,-2],[2,1], [-3,4,2]])
    return MMlow_



def build_corner_MMup_reflect(coord_,direction_,double_B_cell_,double_T_cell_,Cset_cell_,Tset_cell_,Lx,Ly):
    
    Pos=convert_cell_posit(coord_[1-1],coord_[2-1],2,1,direction_,Lx,Ly);
    AA=get_AA_direction(double_B_cell_,double_T_cell_,direction_,Pos);
    Pos=convert_cell_posit(coord_[1-1],coord_[2-1],2,0,direction_,Lx,Ly);
    T1=Tset_cell_[str(Pos[1-1])+','+str(Pos[2-1])]['T'+str(mod1(direction_,4))];
    Pos=convert_cell_posit(coord_[1-1],coord_[2-1],3,0,direction_,Lx,Ly);
    C2=Cset_cell_[str(Pos[1-1])+','+str(Pos[2-1])]['C'+str(mod1(direction_+1,4))];
    Pos=convert_cell_posit(coord_[1-1],coord_[2-1],3,1,direction_,Lx,Ly);
    T2=Tset_cell_[str(Pos[1-1])+','+str(Pos[2-1])]['T'+str(mod1(direction_+1,4))];
    # @tensor MMup_reflect_[:]:=T1[-1,3,1]* C2[1,2]* AA[-2,-4,4,3]* T2[2,4,-3];
    MMup_reflect_ = yastn.ncon([T1, C2, AA, T2], [[-1,3,1], [1,2],[-2,-4,4,3], [2,4,-3]])
    return MMup_reflect_


def build_corner_MMlow_reflect(coord_,direction_,double_B_cell_,double_T_cell_,Cset_cell_,Tset_cell_,Lx,Ly):
 
    Pos=convert_cell_posit(coord_[1-1],coord_[2-1],2,2,direction_,Lx,Ly);
    AA=get_AA_direction(double_B_cell_,double_T_cell_,direction_,Pos);
    Pos=convert_cell_posit(coord_[1-1],coord_[2-1],3,2,direction_,Lx,Ly);
    T2=Tset_cell_[str(Pos[1-1])+','+str(Pos[2-1])]['T'+str(mod1(direction_+1,4))];
    Pos=convert_cell_posit(coord_[1-1],coord_[2-1],2,3,direction_,Lx,Ly);
    T3=Tset_cell_[str(Pos[1-1])+','+str(Pos[2-1])]['T'+str(mod1(direction_-2,4))];
    Pos=convert_cell_posit(coord_[1-1],coord_[2-1],3,3,direction_,Lx,Ly);
    C3=Cset_cell_[str(Pos[1-1])+','+str(Pos[2-1])]['C'+str(mod1(direction_-2,4))];
    # @tensor MMlow_reflect_[:]:=T2[-4,-3,2]*T3[1,-2,-1]*C3[2,1];
    # @tensor MMlow_reflect_[:]:=MMlow_reflect_[-1,1,2,-3]*AA[-2,1,2,-4];
    MMlow_reflect_ = yastn.ncon([T2,T3,C3], [[-4,-3,2], [1,-2,-1],[2,1]])
    MMlow_reflect_ = yastn.ncon([MMlow_reflect_, AA], [[-1,1,2,-3], [-2,1,2,-4]])
    return MMlow_reflect_





def CTM_ite_cell_continuous_update(Cset_cell, Tset_cell, double_B_cell,double_T_cell, chi, direction, ctm_setting, global_args):
    Lx=global_args.Lx;
    Ly=global_args.Ly;
    #println(direction)    
    #
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """


    # cx_cy_matrix=[Lx Ly;Ly Lx;Lx Ly;Ly Lx];
    cx_cy_matrix = numpy.array([
    [Lx, Ly],
    [Ly, Lx],
    [Lx, Ly],
    [Ly, Lx]],dtype=int)
    cx_max=cx_cy_matrix[direction-1,1-1];
    cy_max=cx_cy_matrix[direction-1,2-1];

    for cx in range(1,cx_max+1):

        PM_cell=initial_cell(Lx,Ly);
        PM_inv_cell=initial_cell(Lx,Ly);
        M1tem_cell=initial_cell(Lx,Ly);
        M5tem_cell=initial_cell(Lx,Ly);
        M7tem_cell=initial_cell(Lx,Ly);

        for cy in range(1,cy_max+1):
            coord=[cx,cy];

            ##########################

            MMup=build_corner_MMup(coord,direction,double_B_cell,double_T_cell,Cset_cell,Tset_cell,Lx,Ly);

            MMlow=build_corner_MMlow(coord,direction,double_B_cell,double_T_cell,Cset_cell,Tset_cell,Lx,Ly);

            MMup_reflect=build_corner_MMup_reflect(coord,direction,double_B_cell,double_T_cell,Cset_cell,Tset_cell,Lx,Ly);

            MMlow_reflect=build_corner_MMlow_reflect(coord,direction,double_B_cell,double_T_cell,Cset_cell,Tset_cell,Lx,Ly);

            ############################



            #MMup=permute(MMup,(1,2,),(3,4,))
            #MMup=yastn.transpose(MMup, axes=(0,1,2,3))

            # MMlow=permute(MMlow,(1,2,),(3,4,))
            #MMlow=yastn.transpose(MMlow, axes=(0,1,2,3))

            # MMup_reflect=permute(MMup_reflect,(1,2,),(3,4,))
            #MMup_reflect=yastn.transpose(MMup_reflect, axes=(0,1,2,3))

            # MMlow_reflect=permute(MMlow_reflect,(1,2,),(3,4,))
            #MMlow_reflect=yastn.transpose(MMlow_reflect, axes=(0,1,2,3))

            

            # RMup=permute(MMup*MMup_reflect,(3,4,),(1,2,));
            RMup = yastn.ncon([MMup, MMup_reflect], [[-3,-4,1,2], [1,2,-1,-2]]);
    
            # RMlow=MMlow*MMlow_reflect;
            RMlow = yastn.ncon([MMlow, MMlow_reflect], [[-1,-2,1,2], [1,2,-3,-4]]);

            RMlow_norm=yastn.linalg.norm(RMlow);
            RMlow= RMlow/RMlow_norm;
            RMup_norm=yastn.linalg.norm(RMup);
            RMup= RMup/RMup_norm;

            # M=RMup*RMlow;
            M = yastn.ncon([RMup, RMlow], [[-1,-2,1,2], [1,2,-3,-4]]);


            #####################################


            # uM,sM,vM = my_tsvd(M; trunc=truncdim(chi+chi_extra));
            chi_extra=3;
            uM,sM,vM = yastn.linalg.svd_with_truncation(M, axes=((0, 1), (2, 3)), D_total=chi+chi_extra,svd_on_cpu=True, truncate_multiplets=True, tol=ctm_setting.CTM_trun_tol);

            #println(norm(uM*sM*vM-M)/norm(M));
            #############################################
        


            sM_norm=yastn.linalg.norm(sM);
            sM=sM/sM_norm;
            
            #sM_inv_sqrt=sdiag_inv_sqrt(sM);
            sM_inv_sqrt=sM.rsqrt(cutoff=1e-10);

            # PM_inv=RMlow*vM'*sM_inv_sqrt;
            vMp=vM.conj();
            PM_inv = yastn.ncon([RMlow, vMp, sM_inv_sqrt], [[-1,-2,1,2], [3,1,2], [3,-3]]);
    
            
            # PM=sM_inv_sqrt*uM'*RMup;
            #PM=permute(PM,(2,3,),(1,));
            uMp=uM.conj();
            PM = yastn.ncon([sM_inv_sqrt, uMp, RMup], [[-3,3], [1,2,3], [1,2,-1,-2]]);
            

            Pos=convert_cell_posit(coord[1-1],coord[2-1],0,2,direction, Lx,Ly);
            PM_cell[str(Pos[1-1])+','+str(Pos[2-1])]=PM;
            Pos=convert_cell_posit(coord[1-1],coord[2-1],0,1,direction, Lx,Ly);
            PM_inv_cell[str(Pos[1-1])+','+str(Pos[2-1])]=PM_inv;

        

        for cy in range(1,cy_max+1):
            coord=[cx,cy];

            Pos=convert_cell_posit(coord[1-1],coord[2-1],1,2,direction, Lx,Ly);
            AA=get_AA_direction(double_B_cell,double_T_cell,direction,Pos);
            Pos=convert_cell_posit(coord[1-1],coord[2-1],0,2,direction, Lx,Ly);
            T4=Tset_cell[str(Pos[1-1])+','+str(Pos[2-1])]['T'+str(mod1(direction-1,4))];
            Pos=convert_cell_posit(coord[1-1],coord[2-1],1,0,direction, Lx,Ly);
            T1=Tset_cell[str(Pos[1-1])+','+str(Pos[2-1])]['T'+str(mod1(direction,4))];
            Pos=convert_cell_posit(coord[1-1],coord[2-1],1,3,direction, Lx,Ly);
            T3=Tset_cell[str(Pos[1-1])+','+str(Pos[2-1])]['T'+str(mod1(direction-2,4))];
            Pos=convert_cell_posit(coord[1-1],coord[2-1],0,0,direction, Lx,Ly);
            C1=Cset_cell[str(Pos[1-1])+','+str(Pos[2-1])]['C'+str(mod1(direction,4))];
            Pos=convert_cell_posit(coord[1-1],coord[2-1],0,3,direction, Lx,Ly);
            C4=Cset_cell[str(Pos[1-1])+','+str(Pos[2-1])]['C'+str(mod1(direction-1,4))];


            Posa=convert_cell_posit(coord[1-1],coord[2-1],0,2,direction, Lx,Ly);
            Posb=convert_cell_posit(coord[1-1],coord[2-1],0,2,direction, Lx,Ly);
            # @tensor M5tem[:]:=T4[4,3,1]*AA[3,5,-2,2]*PM_inv_cell[Posa[1-1]][Posa[2-1]][4,5,-1]*PM_cell[Posb[1-1]][Posb[2-1]][1,2,-3];
            M5tem = yastn.ncon([T4, AA, PM_inv_cell[str(Posa[1-1])+','+str(Posa[2-1])], PM_cell[str(Posb[1-1])+','+str(Posb[2-1])]], [[4,3,1], [3,5,-2,2], [4,5,-1],[1,2,-3]]);
            Pos=convert_cell_posit(coord[1-1],coord[2-1],0,0,direction, Lx,Ly);
            #@tensor M1tem[:]:=C1[1,2]*T1[2,3,-2]*PM_inv_cell[Pos[1-1]][Pos[2-1]][1,3,-1];
            M1tem = yastn.ncon([C1, T1, PM_inv_cell[str(Pos[1-1])+','+str(Pos[2-1])]], [[1,2], [2,3,-2], [1,3,-1]]);
            Pos=convert_cell_posit(coord[1-1],coord[2-1],0,3,direction, Lx,Ly);
            #@tensor M7tem[:]:=C4[1,2]*T3[-1,3,1]*PM_cell[Pos[1-1]][Pos[2-1]][2,3,-2];
            M7tem = yastn.ncon([C4, T3, PM_cell[str(Pos[1-1])+','+str(Pos[2-1])]], [[1,2], [-1,3,1], [2,3,-2]]);

            M5tem_norm=yastn.linalg.norm(M5tem);
            M1tem_norm=yastn.linalg.norm(M1tem);
            M7tem_norm=yastn.linalg.norm(M7tem);
    
            M5tem=M5tem/M5tem_norm;
            M1tem=M1tem/M1tem_norm;
            M7tem=M7tem/M7tem_norm;

            Pos=convert_cell_posit(coord[1-1],coord[2-1],1,2,direction, Lx,Ly);
            #M5tem_cell[Pos[1-1]][Pos[2-1]]=M5tem;
            M5tem_cell[str(Pos[1-1])+','+str(Pos[2-1])]=M5tem;
            Pos=convert_cell_posit(coord[1-1],coord[2-1],1,0,direction, Lx,Ly);
            #M1tem_cell[Pos[1-1]][Pos[2-1]]=M1tem;
            M1tem_cell[str(Pos[1-1])+','+str(Pos[2-1])]=M1tem;
            Pos=convert_cell_posit(coord[1-1],coord[2-1],1,3,direction, Lx,Ly);
            #M7tem_cell[Pos[1-1]][Pos[2-1]]=M7tem;
            M7tem_cell[str(Pos[1-1])+','+str(Pos[2-1])]=M7tem;
            
        
        for cy in range(1,cy_max+1):
            coord=[cx,cy];


            Pos=convert_cell_posit(coord[1-1],coord[2-1],1,0,direction, Lx,Ly);
            Cset_cell[str(Pos[1-1])+','+str(Pos[2-1])]['C'+str(mod1(direction,4))]=M1tem_cell[str(Pos[1-1])+','+str(Pos[2-1])];

            Pos=convert_cell_posit(coord[1-1],coord[2-1],1,2,direction, Lx,Ly);
            Tset_cell[str(Pos[1-1])+','+str(Pos[2-1])]['T'+str(mod1(direction-1,4))]=M5tem_cell[str(Pos[1-1])+','+str(Pos[2-1])];

            Pos=convert_cell_posit(coord[1-1],coord[2-1],1,3,direction, Lx,Ly);
            Cset_cell[str(Pos[1-1])+','+str(Pos[2-1])]['C'+str(mod1(direction-1,4))]=M7tem_cell[str(Pos[1-1])+','+str(Pos[2-1])];

            

    return Cset_cell,Tset_cell




def init_CTM_swap(T_double, B_double):

    Cset=initial_Cset();
    Tset=initial_Tset();
    #@tensor AA[:]:=B_double[-1,1,-4]*T_double[-2,-3,1];#(L M U),(D R M) =>(L,D,R,U)

    #@tensor C1[:]:=AA_fused[1,-1,-2,3]*U_L'[2,2,1]*U_U'[3,4,4];#D,R left
    BB1=B_double.unfuse_legs(axes=(0,2));#unfuse L, U=> L',L,M,U',U
    C1 = yastn.ncon([BB1, T_double], [[1,1,3,2,2], [-1,-2,3]]);

    #@tensor C2[:]:=AA_fused[-1,-2,3,1]*U_U'[1,2,2]*U_R'[3,4,4];# L,D left
    BB2=B_double.unfuse_legs(axes=(2));#unfuse U=> L'L,M,U',U
    TT2=T_double.unfuse_legs(axes=(1));#unfuse R=> D'D,R',R,M
    C2 = yastn.ncon([BB2, TT2], [[-1,3,1,1], [-2,2,2,3]]);

    #@tensor C3[:]:=AA_fused[-2,3,1,-1]*U_R'[1,2,2]*U_D'[4,4,3];#U,L left
    TT3=T_double.unfuse_legs(axes=(0,1));#unfuse D R=> D',D,R',R,M
    C3 = yastn.ncon([B_double, TT3], [[-2,3,-1], [1,1,2,2,3]]);

    #@tensor C4[:]:=AA_fused[1,3,-1,-2]*U_L'[2,2,1]*U_D'[4,4,3];#R,U left
    BB4=B_double.unfuse_legs(axes=(0));#unfuse L=> L',L,M,U'U
    TT4=T_double.unfuse_legs(axes=(0));#unfuse D=> D',D,R'R,M
    C4 = yastn.ncon([BB4, TT4], [[2,2,3,-2], [1,1,-1,3]]);

    #(L M U),(D R M) =>(L,D,R,U)

    #@tensor T4[:]:=AA_fused[1,-1,-2,-3]*U_L'[2,2,1];#D,R,U left
    BB5=B_double.unfuse_legs(axes=(0));#unfuse L=> L',L,M,U'U
    T4 = yastn.ncon([BB5, T_double], [[1,1,2,-3], [-1,-2,2]]);

    #@tensor T1[:]:=AA_fused[-1,-2,-3,1]*U_U'[1,2,2];#L,D,R left
    BB6=B_double.unfuse_legs(axes=(2));#unfuse U=> L'L,M,U',U
    T1 = yastn.ncon([BB6, T_double], [[-1,2,1,1], [-2,-3,2]]);

    #@tensor T2[:]:=AA_fused[-2,-3,1,-1]*U_R'[1,2,2];#U,L,D left
    TT7=T_double.unfuse_legs(axes=(1));#unfuse R=> D'D,R',R,M
    T2 = yastn.ncon([B_double, TT7], [[-2,2,-1], [-3,1,1,2]]);

    #@tensor T3[:]:=AA_fused[-3,1,-1,-2]*U_D'[2,2,1];#R,U,L left
    TT8=T_double.unfuse_legs(axes=(0));#unfuse D=> D',D,R'R,M
    T3 = yastn.ncon([B_double, TT8], [[-3,2,-2], [1,1,-1,2]]);

    Cset=OrderedDict();
    Tset=OrderedDict();
    Cset['C1']=C1;
    Cset['C2']=C2;
    Cset['C3']=C3;
    Cset['C4']=C4;
    Tset['T1']=T1;
    Tset['T2']=T2;
    Tset['T3']=T3;
    Tset['T4']=T4;

    # CTM=OrderedDict()
    # CTM['Cset']=Cset;
    # CTM['Tset']=Tset;
    return Cset,Tset
# end

def init_CTM_cell(B_set,T_set,ls_ctm_args, global_args):
    if ls_ctm_args.CTM_ite_info:
        print("initialize CTM from iPESS")
    
    Lx=global_args.Lx;
    Ly=global_args.Ly;


    Cset_cell=initial_cell(Lx,Ly);
    Tset_cell=initial_cell(Lx,Ly);



    for cx in range(1,Lx+1):
        for cy in range(1,Ly+1):
            
            T_double, B_double=build_doublelayer_swap_iPESS(B_set,T_set, [cx,cy]);
            #AA = yastn.ncon([B_double, T_double], [[-1,1, -4], [-2,-3,1]]);

            # CTM_=init_CTM_swap(T_double, B_double);
            Cset,Tset=init_CTM_swap(T_double, B_double);
            Cset_cell[str(cx)+','+str(cy)]=Cset;
            Tset_cell[str(cx)+','+str(cy)]=Tset;
    # CTM_cell=(Cset=Cset_cell,Tset=Tset_cell);
    CTM_cell=OrderedDict()
    CTM_cell['Cset']=Cset_cell;
    CTM_cell['Tset']=Tset_cell;
    return CTM_cell
# end

