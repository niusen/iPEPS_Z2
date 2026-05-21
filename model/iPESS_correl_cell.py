import numpy,torch,math,cmath,copy
import yastn
from collections import OrderedDict
from config.settings import *
from ansatz.triangle_iPESS import *
from ctmrg.Fermionic_CTMRG_unitcell_iPESS import build_double_layer_swap_Tm,build_double_layer_swap_Bm
from torch.utils.checkpoint import checkpoint
from scipy.io import savemat
from model.fermion_ob_iPESS import *




def evaluate_correl(pos,coe,direction, double_B_set,double_T_set, double_B_set_op1,double_T_set_op1, double_B_set_op2,double_T_set_op2, CTM, distance, global_args):
    correl_funs=numpy.zeros((distance,), dtype=numpy.complex128);
    Cset=CTM['Cset'];
    Tset=CTM['Tset'];
    px=pos[1-1];
    py=pos[2-1];
    Lx=global_args.Lx
    Ly=global_args.Ly
    
    if direction=="x":
        C1=Cset[str(mod1(px-1,Lx))+','+str(mod1(py-1,Ly))]['C1'];
        T4=Tset[str(mod1(px-1,Lx))+','+str(mod1(py,Ly))]['T4'];
        C4=Cset[str(mod1(px-1,Lx))+','+str(mod1(py+1,Ly))]['C4'];
        T1=Tset[str(mod1(px,Lx))+','+str(mod1(py-1,Ly))]['T1'];
        T3=Tset[str(mod1(px,Lx))+','+str(mod1(py+1,Ly))]['T3'];
        BB=double_B_set_op1[str(mod1(px,Lx))];
        TT=double_T_set_op1[str(mod1(px,Lx))];
        #@tensor va[:]:=C1[1,3]*T4[2,5,1]*C4[8,2]*T1[3,4,-1]*BB[5,7,4]*TT[6,-2,7]*T3[-3,6,8];
        va = yastn.ncon([C1, T4, C4, T1, BB, TT, T3], [[1,3], [2,5,1], [8,2], [3,4,-1], [5,7,4], [6,-2,7], [-3,6,8]]);

        qx=px+1;
        qy=py;
        T1=Tset[str(mod1(qx,Lx))+','+str(mod1(qy-1,Ly))]['T1'];
        T3=Tset[str(mod1(qx,Lx))+','+str(mod1(qy+1,Ly))]['T3'];
        C2=Cset[str(mod1(qx+1,Lx))+','+str(mod1(qy-1,Ly))]['C2'];
        T2=Tset[str(mod1(qx+1,Lx))+','+str(mod1(qy,Ly))]['T2'];
        C3=Cset[str(mod1(qx+1,Lx))+','+str(mod1(qy+1,Ly))]['C3'];
        BB=double_B_set_op2[str(mod1(qx,Lx))];
        TT=double_T_set_op2[str(mod1(qx,Lx))];
        #@tensor vb[:]:=T1[-1,6,5]*BB[-2,8,6]*TT[4,3,8]*T3[2,4,-3]*C2[5,7]*T2[7,3,1]*C3[1,2];
        vb = yastn.ncon([T1, BB, TT, T3, C2, T2, C3], [[-1,6,5], [-2,8,6], [4,3,8], [2,4,-3], [5,7], [7,3,1], [1,2]]);

        

        #ov=@tensor va[1,2,3]*vb[1,2,3]
        ov = yastn.ncon([va,vb], [[1,2,3], [1,2,3]]).item();
        correl_funs[1-1]=ov;
        
        for dis in range(2,distance+1):
            qx=px+dis-1;
            qy=py;
            T1=Tset[str(mod1(qx,Lx))+','+str(mod1(qy-1,Ly))]['T1'];
            T3=Tset[str(mod1(qx,Lx))+','+str(mod1(qy+1,Ly))]['T3'];
            BB=double_B_set[str(mod1(qx,Lx))];
            TT=double_T_set[str(mod1(qx,Lx))];
            #@tensor va[:]:=va[1,2,6]*T1[1,3,-1]*BB[2,5,3]*TT[4,-2,5]*T3[-3,4,6];
            va = yastn.ncon([va,T1,BB,TT,T3], [[1,2,6], [1,3,-1], [2,5,3], [4,-2,5], [-3,4,6]]);
            va=va*coe;

            qx=px+dis;
            qy=py;
            T1=Tset[str(mod1(qx,Lx))+','+str(mod1(qy-1,Ly))]['T1'];
            T3=Tset[str(mod1(qx,Lx))+','+str(mod1(qy+1,Ly))]['T3'];
            C2=Cset[str(mod1(qx+1,Lx))+','+str(mod1(qy-1,Ly))]['C2'];
            T2=Tset[str(mod1(qx+1,Lx))+','+str(mod1(qy,Ly))]['T2'];
            C3=Cset[str(mod1(qx+1,Lx))+','+str(mod1(qy+1,Ly))]['C3'];
            BB=double_B_set_op2[str(mod1(qx,Lx))];
            TT=double_T_set_op2[str(mod1(qx,Lx))];
            # @tensor vb[:]:=T1[-1,6,5]*BB[-2,8,6]*TT[4,3,8]*T3[2,4,-3]*C2[5,7]*T2[7,3,1]*C3[1,2];
            vb = yastn.ncon([T1, BB, TT, T3, C2, T2, C3], [[-1,6,5], [-2,8,6], [4,3,8], [2,4,-3], [5,7], [7,3,1], [1,2]]);
            # ov=@tensor va[1,2,3]*vb[1,2,3]
            ov = yastn.ncon([va,vb], [[1,2,3], [1,2,3]]).item();
            correl_funs[dis-1]=ov;
        
        return correl_funs
    




def correl_TransOp_x(vl,Tset,py,Lx,Ly):

    for cx in range(1,Lx+1):
        Tup=Tset[str(cx)+','+str(mod1(py,Ly))]['T1'];
        Tdn=Tset[str(cx)+','+str(mod1(py+1,Ly))]['T3'];
        Tup=Tup/(yastn.linalg.norm(Tup))*10;
        Tdn=Tdn/(yastn.linalg.norm(Tdn))*10;
        #@tensor vl[:]:=vl[-1,1,3]*Tup[1,2,-2]*Tdn[-3,2,3];
        vl = yastn.ncon([vl,Tup,Tdn], [[-1,1,3], [1,2,-2],[-3,2,3]]);
    return vl




def solve_correl_length_simple(n_values,CTM_cell,direction, Lx,Ly, config_kwargs, partly):
    Tset=CTM_cell['Tset'];
    local_config_kwargs=dict(config_kwargs)
    local_config_kwargs['default_device']=str(next(iter(Tset.values()))['T1'].device)
    config_Z2 = yastn.make_config(sym='Z2',fermionic=True, **local_config_kwargs)
    Q_set=[];
    if direction=="x":
        if partly:
            y_range=range(1,1+1);
        else:
            y_range=range(1,Ly+1);
        
        eu_cell=[None]*len(y_range);
        for cy in y_range:
            def correl_Trans_fx(x):
                return correl_TransOp_x(x,Tset,cy,Lx,Ly);

            Q_set=[0,1];
            
            eu_set=[None]*len(Q_set);
            for cq in range(0,len(Q_set)):
                Vp=yastn.Leg(config_Z2, s=1, t=[Q_set[cq]], D=[1])
                
                #vl_init = TensorMap(randn, Vp⊗space(Tset[1][mod1(cy,Ly)].T1,1)', space(Tset[1][mod1(cy+1,Ly)].T3,3));
                vl_init=yastn.rand(config=config_Z2, legs=[Vp,Tset['1,'+str(mod1(cy,Ly))]['T1'].get_legs(axes=1-1).conj(), Tset['1,'+str(mod1(cy+1,Ly))]['T3'].get_legs(axes=3-1).conj()]);
                if yastn.linalg.norm(vl_init)>0:
                    vl_init=vl_init/yastn.linalg.norm(vl_init)
                    #eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
                    eu,_=yastn.eigs(correl_Trans_fx, vl_init, k=n_values, which='LM', ncv=10)
                    eu_set[cq]=eu.cpu().numpy();
                else:
                    eu_set[cq]=[];

            eu_cell[cy-1]=eu_set;
    
        return eu_cell,Q_set
    


def build_AA_spin(S1L,S1R,op_string, B_set,T_set,ca,cb,Lx):

    B0=B_set[str(ca)+','+str(cb)];#(LU,M)
    T0=T_set[str(ca)+','+str(cb)];#(M,dRD)
    # @tensor T_new[:]:= T0[-1,1,-3,-4]*S1L[-5,-2,1];#M,d,R,D,virtual
    T_new = yastn.ncon([T0,S1L], [[-1,1,-3,-4], [-5,-2,1]]);
    # U1=unitary(fuse(space(T_new,3)⊗space(T_new,5)), space(T_new,3)⊗space(T_new,5)); 
    # @tensor T_new[:]:=T_new[-1,-2,1,-4,2]*U1[-3,1,2];#M,d,R',D
    T_new=T_new.fuse_legs((0,1,(2,4),3,), mode='hard');
    # T_new=permute(T_new,(1,),(2,3,4,));
    B_double_spin_L = build_double_layer_swap_Tm(B0.conj(),B0, False);#L M U
    T_double_spin_L = build_double_layer_swap_Bm(T0.conj(),T_new, True);#D R M
    

    B0=B_set[str(mod1(ca+1,Lx))+','+str(cb)];#(LU,M)
    T0=T_set[str(mod1(ca+1,Lx))+','+str(cb)];#(M,dRD)
    # Id=unitary(space(S1R,1),space(S1R,1));
    # U12=@ignore_derivatives unitary(fuse(space(B0,3)⊗space(Id,2)), space(B0,3)⊗space(Id,2)); #M
    # @tensor B_new[:]:=B0[1,-2,3]*Id[2,4]*U1'[1,2,-1]*U12[-3,3,4];#L,U,M
    # U2=@ignore_derivatives unitary(fuse(space(T0,3)⊗space(Id,2)), space(T0,3)⊗space(Id,2)); #R
    # @tensor T_new[:]:=T0[1,-2,3,-4]*Id[2,4]*U12'[1,2,-1]*U2[-3,3,4];#M,d,R,D
    B_new = yastn.ncon([B0,op_string], [[-1,-2,-3], [-4,-5]]);#L,U,M,V,Vp
    T_new = yastn.ncon([T0,op_string], [[-1,-2,-3,-4], [-5,-6]]);#M,d,R,D,V,Vp
    B_new=B_new.fuse_legs(((0,3),1,(2,4)), mode='hard');
    T_new=T_new.fuse_legs(((0,4),1,(2,5),3), mode='hard');
    # B_new=permute(B_new,(1,2,),(3,));
    # T_new=permute(T_new,(1,),(2,3,4,));
    B_double_spin_mid= build_double_layer_swap_Tm(B0.conj(),B_new, False);#L M U
    T_double_spin_mid= build_double_layer_swap_Bm(T0.conj(),T_new, True);#D R M
    

    B0=B_set[str(mod1(ca+2,Lx))+','+str(cb)];#(LU,M)
    T0=T_set[str(mod1(ca+2,Lx))+','+str(cb)];#(M,dRD)
    # U23=unitary(fuse(space(B0,3)⊗space(Id,2)), space(B0,3)⊗space(Id,2)); #M
    # @tensor B_new[:]:=B0[1,-2,3]*Id[2,4]*U2'[1,2,-1]*U23[-3,3,4];#L,U,M
    B_new = yastn.ncon([B0,op_string], [[-1,-2,-3], [-4,-5]]);#L,U,M,V,Vp
    B_new=B_new.fuse_legs(((0,3),1,(2,4)), mode='hard');
    # @tensor T_new[:]:= T0[-1,1,-3,-4]*S1R[-5,-2,1];#M,d,R,D,virtual
    # @tensor T_new[:]:=T_new[1,-2,-3,-4,2]*U23'[1,2,-1];#M',d,R,D
    T_new = yastn.ncon([T0,S1R], [[-1,1,-3,-4], [-5,-2,1]]);#M,d,R,D,V
    T_new=T_new.fuse_legs(((0,4),1,2,3), mode='hard');
    # B_new=permute(B_new,(1,2,),(3,));
    # T_new=permute(T_new,(1,),(2,3,4,));
    B_double_spin_R= build_double_layer_swap_Tm(B0.conj(),B_new, False);#L M U
    T_double_spin_R= build_double_layer_swap_Bm(T0.conj(),T_new, True);#D R M


    return B_double_spin_L,T_double_spin_L, B_double_spin_mid,T_double_spin_mid, B_double_spin_R,T_double_spin_R
    


def build_AA_hop(Cdag,C,op_string, B_set,T_set,ca,cb,Lx):
    #the first index of O is dummy

    B0=B_set[str(ca)+','+str(cb)];#(LU,M)
    T0=T_set[str(ca)+','+str(cb)];#(M,dRD)
    # @tensor T_new[:]:= T0[-1,1,-3,-4]*Cdag[-5,-2,1];#M,d,R,D,virtual
    T_new = yastn.ncon([T0,Cdag], [[-1,1,-3,-4], [-5,-2,1]]);#M,d,R,D,V
    # U1=unitary(fuse(space(T_new,3)⊗space(T_new,5)), space(T_new,3)⊗space(T_new,5)); #
    # gate=parity_gate(B0,1); @tensor B_new[:]:=B0[1,-2,-3]*gate[-1,1];#L,U,M
    B_new=yastn.swap_gate(B0, (1-1,1-1));
    # gate=parity_gate(T_new,4); @tensor T_new[:]:=T_new[-1,-2,-3,1,-5]*gate[-4,1];#M,d,R,D,virtual
    T_new=yastn.swap_gate(T_new, (4-1,4-1));
    # gate=parity_gate(B_new,2); @tensor B_new[:]:=B_new[-1,1,-3]*gate[-2,1];
    B_new=yastn.swap_gate(B_new, (2-1,2-1));
    # @tensor T_new[:]:=T_new[-1,-2,1,-4,2]*U1[-3,1,2];#M,d,R',D
    T_new=T_new.fuse_legs((0,1,(2,4),3), mode='hard');
    # B_new=permute(B_new,(1,2,),(3,));
    # T_new=permute(T_new,(1,),(2,3,4,));
    B_double_CdagC_L = build_double_layer_swap_Tm(B0.conj(),B_new, False);#L M U
    T_double_CdagC_L = build_double_layer_swap_Bm(T0.conj(),T_new, True);#D R M

    

    B0=B_set[str(mod1(ca+1,Lx))+','+str(cb)];#(LU,M)
    T0=T_set[str(mod1(ca+1,Lx))+','+str(cb)];#(M,dRD)
    #gate=parity_gate(T0,4); @tensor T_new[:]:=T0[-1,-2,-3,1]*gate[-4,1];#D
    T_new=yastn.swap_gate(T0, (4-1,4-1));
    # U12=unitary(fuse(space(B0,3)⊗space(O_string,2)'), space(B0,3)⊗space(O_string,2)'); 
    #@tensor B_new[:]:=B0[1,-2,3]*O_string[4,2]*U1'[1,2,-1]*U12[-3,3,4];#L,U,M
    # U2=unitary(fuse(space(T_new,3)⊗space(O_string,2)'), space(T_new,3)⊗space(O_string,2)'); 
    #@tensor T_new[:]:=T_new[1,-2,3,-4]*O_string[4,2]*U12'[1,2,-1]*U2[-3,3,4];#M,d,R,D
    B_new = yastn.ncon([B0,op_string], [[-1,-2,-3], [-4,-5]]);#L,U,M,V,Vp
    T_new = yastn.ncon([T_new,op_string], [[-1,-2,-3,-4], [-5,-6]]);#M,d,R,D,V,Vp
    B_new=B_new.fuse_legs(((0,3),1,(2,4)), mode='hard');
    T_new=T_new.fuse_legs(((0,4),1,(2,5),3), mode='hard');
    # B_new=permute(B_new,(1,2,),(3,));
    # T_new=permute(T_new,(1,),(2,3,4,));
    B_double_CdagC_mid = build_double_layer_swap_Tm(B0.conj(),B_new, False);#L M U
    T_double_CdagC_mid = build_double_layer_swap_Bm(T0.conj(),T_new, True);#D R M
    


    B0=B_set[str(mod1(ca+2,Lx))+','+str(cb)];#(LU,M)
    T0=T_set[str(mod1(ca+2,Lx))+','+str(cb)];#(M,dRD)
    # @tensor T_new[:]:= T0[-1,1,-3,-4]*C[-5,-2,1];#M,d,R,D,virtual
    T_new = yastn.ncon([T0,C], [[-1,1,-3,-4], [-5,-2,1]]);
    #gate=parity_gate(B0,1); @tensor B_new[:]:=B0[1,-2,-3]*gate[-1,1];#L
    B_new=yastn.swap_gate(B0, (1-1,1-1));
    #gate=parity_gate(B_new,2); @tensor B_new[:]:=B_new[-1,1,-3]*gate[-2,1];#U
    B_new=yastn.swap_gate(B_new, (2-1,2-1));
    # U23=unitary(fuse(space(B_new,3)⊗space(O_string,2)'), space(B_new,3)⊗space(O_string,2)'); 
    # @tensor B_new[:]:=B_new[1,-2,3]*O_string[4,2]*U2'[1,2,-1]*U23[-3,3,4];#L,U,M
    # @tensor T_new[:]:=T_new[1,-2,-3,-4,2]*U23'[1,2,-1];#M,d,R,D
    B_new = yastn.ncon([B_new,op_string], [[-1,-2,-3], [-4,-5]]);#L,U,M,V,Vp
    B_new=B_new.fuse_legs(((0,3),1,(2,4)), mode='hard');
    T_new=T_new.fuse_legs(((0,4),1,2,3), mode='hard');
    # B_new=permute(B_new,(1,2,),(3,));
    # T_new=permute(T_new,(1,),(2,3,4,));
    B_double_CdagC_R = build_double_layer_swap_Tm(B0.conj(),B_new, False);#L M U
    T_double_CdagC_R = build_double_layer_swap_Bm(T0.conj(),T_new, True);#D R M

    return B_double_CdagC_L,T_double_CdagC_L, B_double_CdagC_mid,T_double_CdagC_mid, B_double_CdagC_R,T_double_CdagC_R


def Cell_take_cy(Cell,cy,Lx,Ly):
    sub_cell=OrderedDict();
    for cx in range(1,Lx+1):
        sub_cell.update({str(cx):Cell[str(cx)+','+str(cy)].clone()})
    return sub_cell


def cal_correl(CTM_cell,B_set,T_set,B_double_set, T_double_set,D,chi,direction,distance, config_kwargs, global_args, partly):
    Lx=global_args.Lx
    Ly=global_args.Ly


    Ident, N_occu, n_double, Cdag, C, CdagC_string =Hamiltonians_spinful_Z2(config_kwargs);
    S1L, S1R, SS_string, chirality_S1, chirality_S2, chirality_S3, chirality_string12, chirality_string23 =Operators_spinful_Z2(config_kwargs);
    

    if partly:
        x_range=range(1,1+1);
        y_range=range(1,1+1);
    else:
        x_range=range(1,Lx+1);
        y_range=range(1,Ly+1);
    

    SS_ob_set=numpy.zeros((len(x_range),len(y_range),distance),dtype=numpy.complex128);
    CdagC_ob_set=numpy.zeros((len(x_range),len(y_range),distance),dtype=numpy.complex128);

    if direction=="x":
        n_values=10;
        eu_x_cell,Q_set=solve_correl_length_simple(n_values,CTM_cell,"x",Lx,Ly,config_kwargs,partly);

        for cb in y_range:
            double_B_spin_L_set=OrderedDict();
            double_T_spin_L_set=OrderedDict();
            double_B_spin_R_set=OrderedDict();
            double_T_spin_R_set=OrderedDict();
            double_B_spin_mid_set=OrderedDict();
            double_T_spin_mid_set=OrderedDict();
            double_B_CdagC_L_set=OrderedDict();
            double_T_CdagC_L_set=OrderedDict();
            double_B_CdagC_R_set=OrderedDict();
            double_T_CdagC_R_set=OrderedDict();
            double_B_CdagC_mid_set=OrderedDict();
            double_T_CdagC_mid_set=OrderedDict();

            for ca in range(1,Lx+1):
            
                B_double_spin_L,T_double_spin_L, B_double_spin_mid,T_double_spin_mid, B_double_spin_R,T_double_spin_R=build_AA_spin(S1L,S1R,SS_string, B_set,T_set,ca,cb,Lx);
                double_B_spin_L_set.update({str(ca):B_double_spin_L});
                double_T_spin_L_set.update({str(ca):T_double_spin_L});
                double_B_spin_mid_set.update({str(mod1(ca+1,Lx)):B_double_spin_mid});
                double_T_spin_mid_set.update({str(mod1(ca+1,Lx)):T_double_spin_mid});
                double_B_spin_R_set.update({str(mod1(ca+2,Lx)):B_double_spin_R});
                double_T_spin_R_set.update({str(mod1(ca+2,Lx)):T_double_spin_R});
                
                B_double_CdagC_L,T_double_CdagC_L, B_double_CdagC_mid,T_double_CdagC_mid, B_double_CdagC_R,T_double_CdagC_R=build_AA_hop(Cdag, C, CdagC_string, B_set,T_set,ca,cb,Lx);
                double_B_CdagC_L_set.update({str(ca):B_double_CdagC_L});
                double_T_CdagC_L_set.update({str(ca):T_double_CdagC_L});
                double_B_CdagC_mid_set.update({str(mod1(ca+1,Lx)):B_double_CdagC_mid});
                double_T_CdagC_mid_set.update({str(mod1(ca+1,Lx)):T_double_CdagC_mid});
                double_B_CdagC_R_set.update({str(mod1(ca+2,Lx)):B_double_CdagC_R});
                double_T_CdagC_R_set.update({str(mod1(ca+2,Lx)):T_double_CdagC_R});
        
            for ca in x_range:
            
            
                #################################
                norms=evaluate_correl([ca,cb],1,"x", Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), CTM_cell, distance,global_args);
                norm_coe=(norms[4+Lx]/norms[4])**(1/Lx); #get a rough normalization coefficient to avoid that the number becomes two small
                norms=evaluate_correl([ca,cb],1/norm_coe,"x", Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), CTM_cell, distance,global_args);
                Spin_ob=evaluate_correl([ca,cb], 1/norm_coe, "x", double_B_spin_mid_set,double_T_spin_mid_set, double_B_spin_L_set,double_T_spin_L_set, double_B_spin_R_set,double_T_spin_R_set, CTM_cell, distance,global_args);
                hopping_ob=evaluate_correl([ca,cb], 1/norm_coe, "x", double_B_CdagC_mid_set,double_T_CdagC_mid_set, double_B_CdagC_L_set,double_T_CdagC_L_set, double_B_CdagC_R_set,double_T_CdagC_R_set, CTM_cell, distance,global_args);
                
                Spin_ob=Spin_ob/norms;
                hopping_ob=hopping_ob/norms;
                # SS_ob_set[str(ca)+','+str(cb)]=Spin_ob;
                # CdagC_ob_set[str(ca)+','+str(cb)]=hopping_ob;

                SS_ob_set[ca-1,cb-1,:]=Spin_ob;
                CdagC_ob_set[ca-1,cb-1,:]=hopping_ob;
            


    mat_filenm="correl_D"+str(D)+"_chi"+str(chi);
    if partly:
        mat_filenm=mat_filenm+"_part";
    else:
        mat_filenm=mat_filenm+"_full";
    


    datadic = {"Lx": Lx, "Ly":Ly, "SS_ob_set": SS_ob_set, "CdagC_ob_set": CdagC_ob_set, "eu_x_cell": eu_x_cell, "Q_set": Q_set}

    savemat(mat_filenm+".mat", datadic)
    return SS_ob_set,CdagC_ob_set



def cal_correl_spin_resolved(CTM_cell,B_set,T_set,B_double_set, T_double_set,D,chi,direction,distance, config_kwargs, global_args, partly):
    Lx=global_args.Lx
    Ly=global_args.Ly

    #Ident, N_occu, n_double, Cdag, C, CdagC_string =Hamiltonians_spinful_Z2(config_kwargs);
    Cdagup_, Cup_, CdagC_up_string, Cdagdn_, Cdn_, CdagC_dn_string=hopping_spin_resolved_Z2(config_kwargs);
    S1L, S1R, SS_string, chirality_S1, chirality_S2, chirality_S3, chirality_string12, chirality_string23 =Operators_spinful_Z2(config_kwargs);
    

    if partly:
        x_range=range(1,1+1);
        y_range=range(1,1+1);
    else:
        x_range=range(1,Lx+1);
        y_range=range(1,Ly+1);
    
    def cal_CdagC(x_range,y_range,distance,direction, Lx,Ly,CTM_cell,config_kwargs,partly,B_set,T_set,B_double_set, T_double_set, Cdag, C, CdagC_string):
        CdagC_ob_set=numpy.zeros((len(x_range),len(y_range),distance),dtype=numpy.complex128);
        if direction=="x":
            n_values=10;
            eu_x_cell,Q_set=solve_correl_length_simple(n_values,CTM_cell,"x",Lx,Ly,config_kwargs,partly);

            for cb in y_range:
                double_B_CdagC_L_set=OrderedDict();
                double_T_CdagC_L_set=OrderedDict();
                double_B_CdagC_R_set=OrderedDict();
                double_T_CdagC_R_set=OrderedDict();
                double_B_CdagC_mid_set=OrderedDict();
                double_T_CdagC_mid_set=OrderedDict();

                for ca in range(1,Lx+1):
                    B_double_CdagC_L,T_double_CdagC_L, B_double_CdagC_mid,T_double_CdagC_mid, B_double_CdagC_R,T_double_CdagC_R=build_AA_hop(Cdag, C, CdagC_string, B_set,T_set,ca,cb,Lx);
                    double_B_CdagC_L_set.update({str(ca):B_double_CdagC_L});
                    double_T_CdagC_L_set.update({str(ca):T_double_CdagC_L});
                    double_B_CdagC_mid_set.update({str(mod1(ca+1,Lx)):B_double_CdagC_mid});
                    double_T_CdagC_mid_set.update({str(mod1(ca+1,Lx)):T_double_CdagC_mid});
                    double_B_CdagC_R_set.update({str(mod1(ca+2,Lx)):B_double_CdagC_R});
                    double_T_CdagC_R_set.update({str(mod1(ca+2,Lx)):T_double_CdagC_R});
            
                for ca in x_range:
                
                    #################################
                    norms=evaluate_correl([ca,cb],1,"x", Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), CTM_cell, distance,global_args);
                    norm_coe=(norms[4+Lx]/norms[4])**(1/Lx); #get a rough normalization coefficient to avoid that the number becomes two small
                    norms=evaluate_correl([ca,cb],1/norm_coe,"x", Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), CTM_cell, distance,global_args);
                    hopping_ob=evaluate_correl([ca,cb], 1/norm_coe, "x", double_B_CdagC_mid_set,double_T_CdagC_mid_set, double_B_CdagC_L_set,double_T_CdagC_L_set, double_B_CdagC_R_set,double_T_CdagC_R_set, CTM_cell, distance,global_args);
                    
                    hopping_ob=hopping_ob/norms;
                    CdagC_ob_set[ca-1,cb-1,:]=hopping_ob;
        return CdagC_ob_set
                
    SS_ob_set=numpy.zeros((len(x_range),len(y_range),distance),dtype=numpy.complex128);

    if direction=="x":
        n_values=10;
        eu_x_cell,Q_set=solve_correl_length_simple(n_values,CTM_cell,"x",Lx,Ly,config_kwargs,partly);

        for cb in y_range:
            double_B_spin_L_set=OrderedDict();
            double_T_spin_L_set=OrderedDict();
            double_B_spin_R_set=OrderedDict();
            double_T_spin_R_set=OrderedDict();
            double_B_spin_mid_set=OrderedDict();
            double_T_spin_mid_set=OrderedDict();

            for ca in range(1,Lx+1):
            
                B_double_spin_L,T_double_spin_L, B_double_spin_mid,T_double_spin_mid, B_double_spin_R,T_double_spin_R=build_AA_spin(S1L,S1R,SS_string, B_set,T_set,ca,cb,Lx);
                double_B_spin_L_set.update({str(ca):B_double_spin_L});
                double_T_spin_L_set.update({str(ca):T_double_spin_L});
                double_B_spin_mid_set.update({str(mod1(ca+1,Lx)):B_double_spin_mid});
                double_T_spin_mid_set.update({str(mod1(ca+1,Lx)):T_double_spin_mid});
                double_B_spin_R_set.update({str(mod1(ca+2,Lx)):B_double_spin_R});
                double_T_spin_R_set.update({str(mod1(ca+2,Lx)):T_double_spin_R});
                
            for ca in x_range:
                #################################
                norms=evaluate_correl([ca,cb],1,"x", Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), CTM_cell, distance,global_args);
                norm_coe=(norms[4+Lx]/norms[4])**(1/Lx); #get a rough normalization coefficient to avoid that the number becomes two small
                norms=evaluate_correl([ca,cb],1/norm_coe,"x", Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), CTM_cell, distance,global_args);
                Spin_ob=evaluate_correl([ca,cb], 1/norm_coe, "x", double_B_spin_mid_set,double_T_spin_mid_set, double_B_spin_L_set,double_T_spin_L_set, double_B_spin_R_set,double_T_spin_R_set, CTM_cell, distance,global_args);

                Spin_ob=Spin_ob/norms;
                SS_ob_set[ca-1,cb-1,:]=Spin_ob;
    CdagC_up_set=cal_CdagC(x_range,y_range,distance,direction, Lx,Ly,CTM_cell,config_kwargs,partly,B_set,T_set,B_double_set, T_double_set, Cdagup_, Cup_, CdagC_up_string);
    CdagC_dn_set=cal_CdagC(x_range,y_range,distance,direction, Lx,Ly,CTM_cell,config_kwargs,partly,B_set,T_set,B_double_set, T_double_set, Cdagdn_, Cdn_, CdagC_dn_string);

    mat_filenm="correl_D"+str(D)+"_chi"+str(chi);
    if partly:
        mat_filenm=mat_filenm+"_part";
    else:
        mat_filenm=mat_filenm+"_full";
    


    datadic = {"Lx": Lx, "Ly":Ly, "SS_ob_set": SS_ob_set, "CdagC_up_set": CdagC_up_set, "CdagC_dn_set": CdagC_dn_set, "eu_x_cell": eu_x_cell, "Q_set": Q_set}

    savemat(mat_filenm+".mat", datadic)
    return SS_ob_set,CdagC_up_set,CdagC_dn_set


def cal_correl_spinless(CTM_cell,B_set,T_set,B_double_set, T_double_set,D,chi,direction,distance, config_kwargs, global_args, partly):
    Lx=global_args.Lx
    Ly=global_args.Ly

    Ident, N_occu, Cdag, C, CdagC_string = Hamiltonians_spinless_Z2(config_kwargs);

    if partly:
        x_range=range(1,1+1);
        y_range=range(1,1+1);
    else:
        x_range=range(1,Lx+1);
        y_range=range(1,Ly+1);

    CdagC_ob_set=numpy.zeros((len(x_range),len(y_range),distance),dtype=numpy.complex128);

    if direction=="x":
        n_values=10;
        eu_x_cell,Q_set=solve_correl_length_simple(n_values,CTM_cell,"x",Lx,Ly,config_kwargs,partly);

        for cb in y_range:
            double_B_CdagC_L_set=OrderedDict();
            double_T_CdagC_L_set=OrderedDict();
            double_B_CdagC_R_set=OrderedDict();
            double_T_CdagC_R_set=OrderedDict();
            double_B_CdagC_mid_set=OrderedDict();
            double_T_CdagC_mid_set=OrderedDict();

            for ca in range(1,Lx+1):
                B_double_CdagC_L,T_double_CdagC_L, B_double_CdagC_mid,T_double_CdagC_mid, B_double_CdagC_R,T_double_CdagC_R=build_AA_hop(Cdag, C, CdagC_string, B_set,T_set,ca,cb,Lx);
                double_B_CdagC_L_set.update({str(ca):B_double_CdagC_L});
                double_T_CdagC_L_set.update({str(ca):T_double_CdagC_L});
                double_B_CdagC_mid_set.update({str(mod1(ca+1,Lx)):B_double_CdagC_mid});
                double_T_CdagC_mid_set.update({str(mod1(ca+1,Lx)):T_double_CdagC_mid});
                double_B_CdagC_R_set.update({str(mod1(ca+2,Lx)):B_double_CdagC_R});
                double_T_CdagC_R_set.update({str(mod1(ca+2,Lx)):T_double_CdagC_R});

            for ca in x_range:
                norms=evaluate_correl([ca,cb],1,"x", Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), CTM_cell, distance,global_args);
                norm_coe=(norms[4+Lx]/norms[4])**(1/Lx);
                norms=evaluate_correl([ca,cb],1/norm_coe,"x", Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), Cell_take_cy(B_double_set,cb,Lx,Ly), Cell_take_cy(T_double_set,cb,Lx,Ly), CTM_cell, distance,global_args);
                hopping_ob=evaluate_correl([ca,cb], 1/norm_coe, "x", double_B_CdagC_mid_set,double_T_CdagC_mid_set, double_B_CdagC_L_set,double_T_CdagC_L_set, double_B_CdagC_R_set,double_T_CdagC_R_set, CTM_cell, distance,global_args);

                hopping_ob=hopping_ob/norms;
                CdagC_ob_set[ca-1,cb-1,:]=hopping_ob;
    else:
        raise NotImplementedError("spinless correlation is currently implemented only for x direction")

    mat_filenm="correl_spinless_D"+str(D)+"_chi"+str(chi);
    if partly:
        mat_filenm=mat_filenm+"_part";
    else:
        mat_filenm=mat_filenm+"_full";

    datadic = {"Lx": Lx, "Ly":Ly, "CdagC_ob_set": CdagC_ob_set, "eu_x_cell": eu_x_cell, "Q_set": Q_set}

    savemat(mat_filenm+".mat", datadic)
    return CdagC_ob_set
