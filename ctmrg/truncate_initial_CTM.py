import yastn
import numpy,torch
import sys
import copy
from collections import OrderedDict
from config.settings import *
from torch.utils.checkpoint import checkpoint
from ansatz.triangle_iPESS import *









def ctm_initial_trun(coord_,direction_,Cset_cell, Tset_cell, chi, ctm_setting, global_args):
    def truncation_f(S):
        return yastn.linalg.truncation_mask_multiplets(S, keep_multiplets=True, D_total=chi, tol=ctm_setting.CTM_trun_tol, tol_block=0.0, eps_multiplet=1.0e-8)
    Lx=global_args.Lx;
    Ly=global_args.Ly;



    Pos=convert_cell_posit(coord_[1-1],coord_[2-1],0,0,direction_,Lx,Ly);
    C1=Cset_cell[str(Pos[1-1])+','+str(Pos[2-1])]['C'+str(mod1(direction_,4))];
    chi_extra=3;
    uM,sM,vM = yastn.linalg.svd_with_truncation(C1, axes=((0,), (1,)), D_total=chi+chi_extra, svd_on_cpu=True, tol=ctm_setting.CTM_trun_tol, mask_f=truncation_f);
    sM_norm=yastn.linalg.norm(sM);
    C1_new=sM/sM_norm;
    Cset_new=update_CTM_C(Cset_cell[str(Pos[1-1])+','+str(Pos[2-1])],C1_new,mod1(direction_,4))
    Cset_cell=update_cell(Cset_cell,Cset_new, Pos[1-1],Pos[2-1],Lx,Ly)

    Pos=convert_cell_posit(coord_[1-1],coord_[2-1],1,0,direction_,Lx,Ly);
    T1=Tset_cell[str(Pos[1-1])+','+str(Pos[2-1])]['T'+str(mod1(direction_,4))];
    T1_new = yastn.ncon([vM,T1], [[-1,1], [1,-2,-3]])
    Tset_new=update_CTM_T(Tset_cell[str(Pos[1-1])+','+str(Pos[2-1])],T1_new,mod1(direction_,4))
    Tset_cell=update_cell(Tset_cell,Tset_new, Pos[1-1],Pos[2-1],Lx,Ly)


    Pos=convert_cell_posit(coord_[1-1],coord_[2-1],0,1,direction_,Lx,Ly);
    T4=Tset_cell[str(Pos[1-1])+','+str(Pos[2-1])]['T'+str(mod1(direction_-1,4))];
    T4_new = yastn.ncon([T4,uM], [[-1,-2,1],[1,-3]])
    Tset_new=update_CTM_T(Tset_cell[str(Pos[1-1])+','+str(Pos[2-1])],T4_new,mod1(direction_-1,4))
    Tset_cell=update_cell(Tset_cell,Tset_new, Pos[1-1],Pos[2-1],Lx,Ly)

    Pos=convert_cell_posit(coord_[1-1],coord_[2-1],0,0,direction_,Lx,Ly);
    T1_left=Tset_cell[str(Pos[1-1])+','+str(Pos[2-1])]['T'+str(mod1(direction_,4))];
    T1_left_new = yastn.ncon([T1_left,vM.conj()], [[-1,-2,1],[-3,1]])
    Tset_new=update_CTM_T(Tset_cell[str(Pos[1-1])+','+str(Pos[2-1])],T1_left_new,mod1(direction_,4))
    Tset_cell=update_cell(Tset_cell,Tset_new, Pos[1-1],Pos[2-1],Lx,Ly)

    Pos=convert_cell_posit(coord_[1-1],coord_[2-1],0,0,direction_,Lx,Ly);
    T4_top=Tset_cell[str(Pos[1-1])+','+str(Pos[2-1])]['T'+str(mod1(direction_-1,4))];
    T4_top_new = yastn.ncon([uM.conj(),T4_top], [[1,-1],[1,-2,-3]])
    Tset_new=update_CTM_T(Tset_cell[str(Pos[1-1])+','+str(Pos[2-1])],T4_top_new,mod1(direction_-1,4))
    Tset_cell=update_cell(Tset_cell,Tset_new, Pos[1-1],Pos[2-1],Lx,Ly)


    return Cset_cell,Tset_cell


