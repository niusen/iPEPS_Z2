import yastn

def mod1(x,y):
    return 1+ (x-1)%y

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
    c1=pos[0];
    c2=pos[1];
    B=B_set[str(c1)+','+str(c2)];
    B_double= build_double_layer_swap_Tm(B.conj(),B, False);#L M U
    T=T_set[str(c1)+','+str(c2)];
    T_double = build_double_layer_swap_Bm(T.conj(),T, True);#D R M
    # @tensor AA[:]:=B_double[-1,1,-4]*T_double[-2,-3,1];#(L M U),(D R M) =>(L,D,R,U)
    # AA = yastn.ncon([B_double, T_double], [[-1,1, -4], [-2,-3,1]]);
    return T_double, B_double



def convert_cell_posit(cx,cy,dx,dy,direction,global_args):

    if direction==1:
        posit=(mod1(cx+dx,global_args.Lx),mod1(cy+dy,global_args.Ly));
    elif direction==2:
        posit=(mod1(cy-dy,global_args.Lx),mod1(cx+dx,global_args.Ly));
    elif direction==3:
        posit=(mod1(cx-dx,global_args.Lx),mod1(cy-dy,global_args.Ly));
    elif direction==4:
        posit=(mod1(cy+dy,global_args.Lx),mod1(cx-dx,global_args.Ly));
    return posit


def rotate_AA_direction(AA_fused,direction):
    #AA_rotated=permute(AA_fused, (mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),),());
    AA_rotated=yastn.transpose(AA_rotated, axes=(mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4)))
    return AA_rotated




def Fermionic_CTMRG_cell_iPESS(B_set,T_set,A_cell,chi,init,CTM0, global_args,ctm_setting):

    #Ref: PHYSICAL REVIEW B 98, 235148 (2018)
    ########################
    CTM_trun_tol=ctm_setting.CTM_trun_tol;
    CTM_ite_info=ctm_setting.CTM_ite_info;
    CTM_conv_info=ctm_setting.CTM_conv_info;
    projector_strategy=ctm_setting.projector_strategy;
    CTM_trun_svd=ctm_setting.CTM_trun_svd;
    svd_lanczos_tol=ctm_setting.svd_lanczos_tol;
    CTM_ite_nums=ctm_setting.CTM_ite_nums;
    construct_double_layer=ctm_setting.construct_double_layer;
    #######################
    if (CTM_trun_svd==true) & (projector_strategy=="4x4")
        println("Attention: truncated svd with 4x4 projector could give large error");
    end

    #initial corner transfer matrix
    if init.reconstruct_AA
        double_B_cell=initial_tuple_cell(Lx,Ly);
        double_T_cell=initial_tuple_cell(Lx,Ly);
        U_L_cell=initial_tuple_cell(Lx,Ly);
        U_D_cell=initial_tuple_cell(Lx,Ly);
        U_R_cell=initial_tuple_cell(Lx,Ly);
        U_U_cell=initial_tuple_cell(Lx,Ly);
        for cx=1:Lx
            for cy=1:Ly
                T_double, B_double, U_L_,U_D_,U_R_,U_U_=build_doublelayer_swap_iPESS(B_set,T_set, [cx,cy]);

                double_B_cell=fill_tuple(double_B_cell, B_double, cx,cy);
                double_T_cell=fill_tuple(double_T_cell, T_double, cx,cy);
                U_L_cell=fill_tuple(U_L_cell, U_L_, cx,cy);
                U_D_cell=fill_tuple(U_D_cell, U_D_, cx,cy);
                U_R_cell=fill_tuple(U_R_cell, U_R_, cx,cy);
                U_U_cell=fill_tuple(U_U_cell, U_U_, cx,cy);
            end
        end
        AA_memory=@ignore_derivatives (Base.summarysize(double_B_cell)+Base.summarysize(double_T_cell))/1024/1024;
        @ignore_derivatives if CTM_ite_info
            println("Memory cost of double layer iPESS: "*string(AA_memory)*" Mb.");flush(stdout);
        end
    else
        # AA_fused_cell=auxi_tensors.AA_fused_cell;
        # U_L_cell=auxi_tensors.U_L_cell;
        # U_D_cell=auxi_tensors.U_D_cell;
        # U_R_cell=auxi_tensors.U_R_cell;
        # U_U_cell=auxi_tensors.U_U_cell;
    end

    if init.reconstruct_CTM
        CTM_cell= init_CTM_cell(chi,B_set,T_set,  init.init_type,CTM_ite_info);
    else
        CTM_cell=deepcopy(CTM0);
    end
    
    ss_old1_cell= Matrix(undef,Lx,Ly);
    ss_old2_cell= Matrix(undef,Lx,Ly);
    ss_old3_cell= Matrix(undef,Lx,Ly);
    ss_old4_cell= Matrix(undef,Lx,Ly);
    ss_new1_cell= Matrix(undef,Lx,Ly);
    ss_new2_cell= Matrix(undef,Lx,Ly);
    ss_new3_cell= Matrix(undef,Lx,Ly);
    ss_new4_cell= Matrix(undef,Lx,Ly);
    er1_cell= Matrix(undef,Lx,Ly);
    er2_cell= Matrix(undef,Lx,Ly);
    er3_cell= Matrix(undef,Lx,Ly);
    er4_cell= ones(Lx,Ly);


    Cset_cell=CTM_cell.Cset;
    Tset_cell=CTM_cell.Tset;
    conv_check="singular_value"

    @ignore_derivatives for cx=1:Lx
        for cy=1:Ly
            ss_old1_cell[cx,cy]=ones(chi)*2;
            ss_old2_cell[cx,cy]=ones(chi)*2;
            ss_old3_cell[cx,cy]=ones(chi)*2;
            ss_old4_cell[cx,cy]=ones(chi)*2;
        end
    end
    d=2;
    rho_old=Matrix(I,d^3,d^3);

    #Iteration

    print_corner=false;
    C1_spec_cell=@ignore_derivatives Matrix(undef,Lx,Ly);
    C2_spec_cell=@ignore_derivatives Matrix(undef,Lx,Ly);
    C3_spec_cell=@ignore_derivatives Matrix(undef,Lx,Ly);
    C4_spec_cell=@ignore_derivatives Matrix(undef,Lx,Ly);
    @ignore_derivatives if print_corner
        for cx=1:Lx
            for cy=1:Ly
                println("cell position: "*string([cx,cy]))
                println("corner 4:")
                C4_spec=svdvals(convert(Array,Cset[4][cx,cy]));
                C4_spec_cell[cx,cy]=C4_spec/C4_spec[1];
                println(C4_spec);
                println("corner 1:")
                C1_spec=svdvals(convert(Array,Cset[1][cx,cy]));
                C1_spec_cell[cx,cy]=C1_spec/C1_spec[1];
                println(C1_spec);
                println("corner 3:")
                C3_spec=svdvals(convert(Array,Cset[3][cx,cy]));
                C3_spec_cell[cx,cy]=C3_spec/C3_spec[1];
                println(C3_spec);
                println("corner 2:")
                C2_spec=svdvals(convert(Array,Cset[2][cx,cy]));
                C2_spec_cell[cx,cy]=C2_spec/C2_spec[1];
                println(C2_spec);
            end
        end
        println("CTM init finished")
    end
    


    @ignore_derivatives if CTM_ite_info
        println("start CTM iterations:")
    end
    ite_num=0;
    ite_err=1;
    err_set=1;
    

    CTM_ite_cell=CTM_ite_cell_continuous_update;
    

    for ci=1:CTM_ite_nums
        ite_num=ci;
        #direction_order=[1,2,3,4];
        #direction_order=[4,1,2,3];
        direction_order=[3,4,1,2];
        @time begin
            for direction in direction_order
                Cset_cell,Tset_cell=CTM_ite_cell(Cset_cell, Tset_cell, double_B_cell,double_T_cell, chi, direction,CTM_trun_tol,CTM_ite_info,projector_strategy,CTM_trun_svd,svd_lanczos_tol,construct_double_layer);
            end
        end

        print_corner=false;
        @ignore_derivatives if print_corner
            for cx=1:Lx
                for cy=1:Ly
                    println("cell position: "*string([cx,cy]))
                    println("corner 4:")
                    C4_spec=svdvals(convert(Array,Cset[4][cx,cy]));
                    C4_spec_cell[cx,cy]=C4_spec/C4_spec[1];
                    println(C4_spec);
                    println("corner 1:")
                    C1_spec=svdvals(convert(Array,Cset[1][cx,cy]));
                    C1_spec_cell[cx,cy]=C1_spec/C1_spec[1];
                    println(C1_spec);
                    println("corner 3:")
                    C3_spec=svdvals(convert(Array,Cset[3][cx,cy]));
                    C3_spec_cell[cx,cy]=C3_spec/C3_spec[1];
                    println(C3_spec);
                    println("corner 2:")
                    C2_spec=svdvals(convert(Array,Cset[2][cx,cy]));
                    C2_spec_cell[cx,cy]=C2_spec/C2_spec[1];
                    println(C2_spec);
                end
            end
            println("next iteration:")
        end
        


        if conv_check=="singular_value" #check convergence of singular value
            for cx=1:Lx
                for cy=1:Ly
                    er1,ss_new1=@ignore_derivatives spectrum_conv_check(ss_old1_cell[cx,cy],Cset_cell[cx][cy].C1);
                    er2,ss_new2=@ignore_derivatives spectrum_conv_check(ss_old2_cell[cx,cy],Cset_cell[cx][cy].C2);
                    er3,ss_new3=@ignore_derivatives spectrum_conv_check(ss_old3_cell[cx,cy],Cset_cell[cx][cy].C3);
                    er4,ss_new4=@ignore_derivatives spectrum_conv_check(ss_old4_cell[cx,cy],Cset_cell[cx][cy].C4);

                    # println([cx,cy])
                    # println(ss_new1)
                    # println(ss_new2)
                    # println(ss_new3)
                    # println(ss_new4)
                    

                    @ignore_derivatives er1_cell[cx,cy]=er1;
                    @ignore_derivatives er2_cell[cx,cy]=er2;
                    @ignore_derivatives er3_cell[cx,cy]=er3;
                    @ignore_derivatives er4_cell[cx,cy]=er4;
                    @ignore_derivatives ss_new1_cell[cx,cy]=ss_new1;
                    @ignore_derivatives ss_new2_cell[cx,cy]=ss_new2;
                    @ignore_derivatives ss_new3_cell[cx,cy]=ss_new3;
                    @ignore_derivatives ss_new4_cell[cx,cy]=ss_new4;

                end
            end

            er=@ignore_derivatives maximum([maximum(er1_cell[:]),maximum(er2_cell[:]),maximum(er3_cell[:]),maximum(er4_cell[:])]);
            err_set=vcat(err_set,er);

            ite_err=er;
            if CTM_ite_info
                println("CTMRG iteration: "*string(ci)*", CTMRG err: "*string(er));flush(stdout);
            end
            if er<ctm_setting.CTM_conv_tol
                break;
            end

            if ci>30
                err_recent=err_set[end-10:end];
                Std=std(err_recent)/mean(err_recent);
                if (Std<0.001)&(er>1e-4)
                    break;
                end

            end

            ss_old1_cell=ss_new1_cell;
            ss_old2_cell=ss_new2_cell;
            ss_old3_cell=ss_new3_cell;
            ss_old4_cell=ss_new4_cell;
        elseif conv_check=="density_matrix" #check reduced density matrix
        end
    end

    CTM_cell=(Cset=Cset_cell,Tset=Tset_cell);
    if CTM_conv_info
        return CTM_cell, double_B_cell,double_T_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err
    else
        return CTM_cell, double_B_cell,double_T_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell
    # end

# end

def get_AA_direction(double_B_cell,double_T_cell,direction,pos):
    B_double=double_B_cell[pos[1]][pos[2]];
    T_double=double_T_cell[pos[1]][pos[2]];
    @tensor AA[:]:=B_double[-1,1,-4]*T_double[-2,-3,1];
    return rotate_AA_direction(AA,direction)
end


def build_corner_MMup(coord_,direction_,double_B_cell_,double_T_cell_,Cset_cell_,Tset_cell_,Lx,Ly):
    global Lx,Ly
    Pos=convert_cell_posit(coord_[1],coord_[2],1,1,direction_);
    AA=get_AA_direction(double_B_cell_,double_T_cell_,direction_,Pos);
    Pos=convert_cell_posit(coord_[1],coord_[2],0,0,direction_);
    C1=get_Cset(Cset_cell_[Pos[1]][Pos[2]], mod1(direction_,4));
    Pos=convert_cell_posit(coord_[1],coord_[2],1,0,direction_);
    T1=get_Tset(Tset_cell_[Pos[1]][Pos[2]], mod1(direction_,4));
    Pos=convert_cell_posit(coord_[1],coord_[2],0,1,direction_);
    T4=get_Tset(Tset_cell_[Pos[1]][Pos[2]], mod1(direction_-1,4));
    @tensor MMup_[:]:=C1[1,2]*T1[2,3,-3]*T4[-1,4,1]*AA[4,-2,-4,3];
    return MMup_
# end

def build_corner_MMlow(coord_,direction_,double_B_cell_,double_T_cell_,Cset_cell_,Tset_cell_,Lx,Ly):
    global Lx,Ly
    Pos=convert_cell_posit(coord_[1],coord_[2],1,2,direction_);
    AA=get_AA_direction(double_B_cell_,double_T_cell_,direction_,Pos);
    Pos=convert_cell_posit(coord_[1],coord_[2],0,2,direction_);
    T4=get_Tset(Tset_cell_[Pos[1]][Pos[2]], mod1(direction_-1,4));
    Pos=convert_cell_posit(coord_[1],coord_[2],0,3,direction_);
    C4=get_Cset(Cset_cell_[Pos[1]][Pos[2]], mod1(direction_-1,4));
    Pos=convert_cell_posit(coord_[1],coord_[2],1,3,direction_);
    T3=get_Tset(Tset_cell_[Pos[1]][Pos[2]], mod1(direction_-2,4));
    @tensor MMlow_[:]:=T4[1,3,-1]*AA[3,4,-4,-2]*C4[2,1]*T3[-3,4,2];
    return MMlow_
# end


def build_corner_MMup_reflect(coord_,direction_,double_B_cell_,double_T_cell_,Cset_cell_,Tset_cell_,Lx,Ly):
    global Lx,Ly
    Pos=convert_cell_posit(coord_[1],coord_[2],2,1,direction_);
    AA=get_AA_direction(double_B_cell_,double_T_cell_,direction_,Pos);
    Pos=convert_cell_posit(coord_[1],coord_[2],2,0,direction_);
    T1=get_Tset(Tset_cell_[Pos[1]][Pos[2]], mod1(direction_,4));
    Pos=convert_cell_posit(coord_[1],coord_[2],3,0,direction_);
    C2=get_Cset(Cset_cell_[Pos[1]][Pos[2]], mod1(direction_+1,4));
    Pos=convert_cell_posit(coord_[1],coord_[2],3,1,direction_);
    T2=get_Tset(Tset_cell_[Pos[1]][Pos[2]], mod1(direction_+1,4));
    @tensor MMup_reflect_[:]:=T1[-1,3,1]* C2[1,2]* AA[-2,-4,4,3]* T2[2,4,-3];
    return MMup_reflect_
# end

def build_corner_MMlow_reflect(coord_,direction_,double_B_cell_,double_T_cell_,Cset_cell_,Tset_cell_,Lx,Ly):
    global Lx,Ly
    Pos=convert_cell_posit(coord_[1],coord_[2],2,2,direction_);
    AA=get_AA_direction(double_B_cell_,double_T_cell_,direction_,Pos);
    Pos=convert_cell_posit(coord_[1],coord_[2],3,2,direction_);
    T2=get_Tset(Tset_cell_[Pos[1]][Pos[2]], mod1(direction_+1,4));
    Pos=convert_cell_posit(coord_[1],coord_[2],2,3,direction_);
    T3=get_Tset(Tset_cell_[Pos[1]][Pos[2]], mod1(direction_-2,4));
    Pos=convert_cell_posit(coord_[1],coord_[2],3,3,direction_);
    C3=get_Cset(Cset_cell_[Pos[1]][Pos[2]], mod1(direction_-2,4));
    @tensor MMlow_reflect_[:]:=T2[-4,-3,2]*T3[1,-2,-1]*C3[2,1];
    @tensor MMlow_reflect_[:]:=MMlow_reflect_[-1,1,2,-3]*AA[-2,1,2,-4];
    return MMlow_reflect_
# end




def CTM_ite_cell_continuous_update(Cset_cell, Tset_cell, double_B_cell,double_T_cell, chi, direction, trun_tol,CTM_ite_info,projector_strategy,CTM_trun_svd,svd_lanczos_tol,construct_double_layer):
    global Lx,Ly
    #println(direction)    
    #
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """

    # if (direction==1)|(direction==3)
    #     cx_max=Lx;
    #     cy_max=Ly;
    # elseif (direction==2)|(direction==4)
    #     cx_max=Ly;
    #     cy_max=Lx;
    # end
    cx_cy_matrix=[Lx Ly;Ly Lx;Lx Ly;Ly Lx];
    cx_max=cx_cy_matrix[direction,1];
    cy_max=cx_cy_matrix[direction,2];

    for cx=1:cx_max

        PM_cell=initial_tuple_cell(Lx,Ly);
        PM_inv_cell=initial_tuple_cell(Lx,Ly);
        M1tem_cell=initial_tuple_cell(Lx,Ly);
        M5tem_cell=initial_tuple_cell(Lx,Ly);
        M7tem_cell=initial_tuple_cell(Lx,Ly);

        for cy=1:cy_max
            coord=[cx,cy];

            ##########################
            parall_data=[];
            parall_data=@sync @distributed  (append_data)  for ccc=1:4
                BLAS.set_num_threads(5);
                if ccc==1
                    MM=build_corner_MMup(coord,direction,double_B_cell,double_T_cell,Cset_cell,Tset_cell,Lx,Ly);
                elseif ccc==2
                    MM=build_corner_MMlow(coord,direction,double_B_cell,double_T_cell,Cset_cell,Tset_cell,Lx,Ly);
                elseif ccc==3
                    MM=build_corner_MMup_reflect(coord,direction,double_B_cell,double_T_cell,Cset_cell,Tset_cell,Lx,Ly);
                elseif ccc==4
                    MM=build_corner_MMlow_reflect(coord,direction,double_B_cell,double_T_cell,Cset_cell,Tset_cell,Lx,Ly);
                end
                [(myid(), ccc, MM)]
            end

            for ccc=1:4
                if parall_data[ccc][2]==1
                    MMup=parall_data[ccc][3];
                elseif parall_data[ccc][2]==2
                    MMlow=parall_data[ccc][3];
                elseif parall_data[ccc][2]==3
                    MMup_reflect=parall_data[ccc][3];
                elseif parall_data[ccc][2]==4
                    MMlow_reflect=parall_data[ccc][3];
                end
            end
            parall_data=[];
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
        


            sM_norm=yastn.linalg.normnorm(sM);
            sM=sM/sM_norm;
            
            #sM_inv_sqrt=sdiag_inv_sqrt(sM);
            sM_inv_sqrt=sM.rsqrt(cutoff=1e-10);

            PM_inv=RMlow*vM'*sM_inv_sqrt;
    
            PM=sM_inv_sqrt*uM'*RMup;
            PM=permute(PM,(2,3,),(1,));

            Pos=convert_cell_posit(coord[1],coord[2],0,2,direction);
            #PM_cell[Pos[1]][Pos[2]]=PM;
            PM_cell=fill_tuple(PM_cell,PM, Pos[1],Pos[2])
            Pos=convert_cell_posit(coord[1],coord[2],0,1,direction);
            #PM_inv_cell[Pos[1]][Pos[2]]=PM_inv;
            PM_inv_cell=fill_tuple(PM_inv_cell,PM_inv, Pos[1],Pos[2])

        end

        for cy=1:cy_max
            coord=[cx,cy];

            Pos=convert_cell_posit(coord[1],coord[2],1,2,direction);
            AA=get_AA_direction(double_B_cell,double_T_cell,direction,Pos);;
            Pos=convert_cell_posit(coord[1],coord[2],0,2,direction);
            T4=get_Tset(Tset_cell[Pos[1]][Pos[2]], mod1(direction-1,4));
            Pos=convert_cell_posit(coord[1],coord[2],1,0,direction);
            T1=get_Tset(Tset_cell[Pos[1]][Pos[2]], mod1(direction,4));
            Pos=convert_cell_posit(coord[1],coord[2],1,3,direction);
            T3=get_Tset(Tset_cell[Pos[1]][Pos[2]], mod1(direction-2,4));
            Pos=convert_cell_posit(coord[1],coord[2],0,0,direction);
            C1=get_Cset(Cset_cell[Pos[1]][Pos[2]], mod1(direction,4));
            Pos=convert_cell_posit(coord[1],coord[2],0,3,direction);
            C4=get_Cset(Cset_cell[Pos[1]][Pos[2]], mod1(direction-1,4));


            Posa=convert_cell_posit(coord[1],coord[2],0,2,direction);
            Posb=convert_cell_posit(coord[1],coord[2],0,2,direction);
            @tensor M5tem[:]:=T4[4,3,1]*AA[3,5,-2,2]*PM_inv_cell[Posa[1]][Posa[2]][4,5,-1]*PM_cell[Posb[1]][Posb[2]][1,2,-3];
            Pos=convert_cell_posit(coord[1],coord[2],0,0,direction);
            @tensor M1tem[:]:=C1[1,2]*T1[2,3,-2]*PM_inv_cell[Pos[1]][Pos[2]][1,3,-1];
            Pos=convert_cell_posit(coord[1],coord[2],0,3,direction);
            @tensor M7tem[:]:=C4[1,2]*T3[-1,3,1]*PM_cell[Pos[1]][Pos[2]][2,3,-2];

            M5tem_norm=yastn.linalg.norm(M5tem);
            M1tem_norm=yastn.linalg.norm(M1tem);
            M7tem_norm=yastn.linalg.norm(M7tem);
    
            M5tem=M5tem/M5tem_norm;
            M1tem=M1tem/M1tem_norm;
            M7tem=M7tem/M7tem_norm;

            Pos=convert_cell_posit(coord[1],coord[2],1,2,direction);
            #M5tem_cell[Pos[1]][Pos[2]]=M5tem;
            M5tem_cell=fill_tuple(M5tem_cell, M5tem, Pos[1],Pos[2]);
            Pos=convert_cell_posit(coord[1],coord[2],1,0,direction);
            #M1tem_cell[Pos[1]][Pos[2]]=M1tem;
            M1tem_cell=fill_tuple(M1tem_cell, M1tem, Pos[1],Pos[2]);
            Pos=convert_cell_posit(coord[1],coord[2],1,3,direction);
            #M7tem_cell[Pos[1]][Pos[2]]=M7tem;
            M7tem_cell=fill_tuple(M7tem_cell, M7tem, Pos[1],Pos[2]);
            
        end


        for cy=1:cy_max
            coord=[cx,cy];


            Pos=convert_cell_posit(coord[1],coord[2],1,0,direction);
            #Cset_cell[Pos[1]][Pos[2]][mod1(direction,4)]=M1tem_cell[Pos[1]][Pos[2]];
            Cset_old=Cset_cell[Pos[1]][Pos[2]];
            Cset_new=set_Cset(Cset_old, M1tem_cell[Pos[1]][Pos[2]], mod1(direction,4))
            Cset_cell=fill_tuple(Cset_cell, Cset_new, Pos[1],Pos[2]);

            Pos=convert_cell_posit(coord[1],coord[2],1,2,direction);
            #Tset_cell[Pos[1]][Pos[2]][mod1(direction-1,4)]=M5tem_cell[Pos[1]][Pos[2]];
            Tset_old=Tset_cell[Pos[1]][Pos[2]];
            Tset_new=set_Tset(Tset_old, M5tem_cell[Pos[1]][Pos[2]], mod1(direction-1,4))
            Tset_cell=fill_tuple(Tset_cell, Tset_new, Pos[1],Pos[2]);

            Pos=convert_cell_posit(coord[1],coord[2],1,3,direction);
            #Cset_cell[Pos[1]][Pos[2]][mod1(direction-1,4)]=M7tem_cell[Pos[1]][Pos[2]];
            Cset_old=Cset_cell[Pos[1]][Pos[2]];
            Cset_new=set_Cset(Cset_old, M7tem_cell[Pos[1]][Pos[2]], mod1(direction-1,4))
            Cset_cell=fill_tuple(Cset_cell, Cset_new, Pos[1],Pos[2]);
            
        end
    end
    return Cset_cell,Tset_cell
# end



def init_CTM_swap(chi,A,AA_fused, U_L,U_D,U_R,U_U,type,CTM_ite_info):
    @ignore_derivatives  if CTM_ite_info
        display("initialize CTM")
    end

    CTM=[];

    Cset=Cset_struc(A,A,A,A)
    Tset=Tset_struc(A,A,A,A)

    
    if type=="PBC"

        @tensor C1[:]:=AA_fused[1,-1,-2,3]*U_L'[2,2,1]*U_U'[3,4,4];
        @tensor C2[:]:=AA_fused[-1,-2,3,1]*U_U'[1,2,2]*U_R'[3,4,4];
        @tensor C3[:]:=AA_fused[-2,3,1,-1]*U_R'[1,2,2]*U_D'[4,4,3];
        @tensor C4[:]:=AA_fused[1,3,-1,-2]*U_L'[2,2,1]*U_D'[4,4,3];

        @tensor T4[:]:=AA_fused[1,-1,-2,-3]*U_L'[2,2,1];
        @tensor T1[:]:=AA_fused[-1,-2,-3,1]*U_U'[1,2,2];
        @tensor T2[:]:=AA_fused[-2,-3,1,-1]*U_R'[1,2,2];
        @tensor T3[:]:=AA_fused[-3,1,-1,-2]*U_D'[2,2,1];

        Cset=Cset_struc(C1,C2,C3,C4);
        Tset=Tset_struc(T1,T2,T3,T4);
    
        CTM=CTM_struc(Cset,Tset)

    elseif type=="random"
    end

    CTM=CTM_struc(Cset, Tset);
    return CTM
# end

def init_CTM_cell(chi,B_set,T_set, type,CTM_ite_info):
    @ignore_derivatives  if CTM_ite_info
        display("initialize CTM from iPESS")
    end
    global Lx,Ly




    Cset_cell=initial_tuple_cell(Lx,Ly);
    Tset_cell=initial_tuple_cell(Lx,Ly);

    if type=="PBC"
        for cx=1:Lx
            for cy=1:Ly
                T_double, B_double=build_doublelayer_swap_iPESS(B_set,T_set, [cx+1,cy+1]);
                #AA = yastn.ncon([B_double, T_double], [[-1,1, -4], [-2,-3,1]]);
                small_tensor=B_set[1,1];#the value not important

                CTM_=init_CTM_swap(chi,small_tensor,AA, type,false);
                Cset_cell=fill_tuple(Cset_cell,CTM_.Cset,cx,cy);
                Tset_cell=fill_tuple(Tset_cell,CTM_.Tset,cx,cy);
            end
        end
        CTM_cell=(Cset=Cset_cell,Tset=Tset_cell);
        return CTM_cell
    elseif type=="random"
    end
# end

