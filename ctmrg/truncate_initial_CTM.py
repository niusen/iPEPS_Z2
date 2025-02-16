import yastn
import numpy,torch
import sys
import copy
from collections import OrderedDict
from config.settings import *
from torch.utils.checkpoint import checkpoint
from ansatz.triangle_iPESS import *









def ctm_update_single_cx(Cset_cell, Tset_cell, chi, ctm_setting, global_args):
    def truncation_f(S):
        return yastn.linalg.truncation_mask_multiplets(S, keep_multiplets=True, D_total=chi, tol=ctm_setting.CTM_trun_tol, tol_block=0.0, eps_multiplet=1.0e-8)
    Lx=global_args.Lx;
    Ly=global_args.Ly;
    PM_cell=initial_cell(Lx,Ly);
    PM_inv_cell=initial_cell(Lx,Ly);
    M1tem_cell=initial_cell(Lx,Ly);
    M5tem_cell=initial_cell(Lx,Ly);
    M7tem_cell=initial_cell(Lx,Ly);

    for cx in range(1,Lx+1):
        for cy in range(1,Ly+1):
            coord=[cx,cy];
            print(coord)

            ##########################


            # uM,sM,vM = my_tsvd(M; trunc=truncdim(chi+chi_extra));
            chi_extra=3;
            


            # uM,sM,vM = yastn.linalg.svd_with_truncation(M, axes=((0, 1), (2, 3)), D_total=chi+chi_extra,svd_on_cpu=True, truncate_multiplets=True, tol=ctm_setting.CTM_trun_tol);
            uM,sM,vM = yastn.linalg.svd_with_truncation(M, axes=((0, 1), (2, 3)), D_total=chi+chi_extra, svd_on_cpu=True, tol=ctm_setting.CTM_trun_tol, mask_f=truncation_f);

            

            #############################################



            sM_norm=yastn.linalg.norm(sM);
            sM=sM/sM_norm;
            
            #sM_inv_sqrt=sdiag_inv_sqrt(sM);
            # sM_inv_sqrt=sM.rsqrt(cutoff=1e-10);
            sM_inv_sqrt=sM.rsqrt(cutoff=ctm_setting.CTM_trun_tol);
            # sM_inv_sqrt=sM_inv_sqrt.rsqrt(cutoff=1e-10);
            #sM_inv_sqrt_1d,bb=yastn.Tensor.compress_to_1d(sM_inv_sqrt);
            #print(sM_inv_sqrt_1d)

            # PM_inv=RMlow*vM'*sM_inv_sqrt;
            vMp=vM.conj();
            if ctm_setting.doublelayer_on_cpu:#send back to gpu 
                PM_inv = yastn.ncon([RMlow.to(global_args.device), vMp, sM_inv_sqrt], [[-1,-2,1,2], [3,1,2], [3,-3]]);
            else:
                PM_inv = yastn.ncon([RMlow, vMp, sM_inv_sqrt], [[-1,-2,1,2], [3,1,2], [3,-3]]);

            
            # PM=sM_inv_sqrt*uM'*RMup;
            #PM=permute(PM,(2,3,),(1,));
            uMp=uM.conj();
            if ctm_setting.doublelayer_on_cpu:#send back to gpu 
                PM = yastn.ncon([sM_inv_sqrt, uMp, RMup.to(global_args.device)], [[-3,3], [1,2,3], [1,2,-1,-2]]);
            else:
                PM = yastn.ncon([sM_inv_sqrt, uMp, RMup], [[-3,3], [1,2,3], [1,2,-1,-2]]);

            

            Pos=convert_cell_posit(coord[1-1],coord[2-1],0,2,direction, Lx,Ly);
            PM_cell[str(Pos[1-1])+','+str(Pos[2-1])]=PM;
            Pos=convert_cell_posit(coord[1-1],coord[2-1],0,1,direction, Lx,Ly);
            PM_inv_cell[str(Pos[1-1])+','+str(Pos[2-1])]=PM_inv;

        


            coord=[cx,cy];
            #PM_cell, PM_inv_cell, M1tem_cell, M5tem_cell, M7tem_cell=checkpoint(prepare_update, Cset_cell, Tset_cell, double_B_cell,double_T_cell, PM_cell, PM_inv_cell, M1tem_cell, M5tem_cell, M7tem_cell, coord,direction,Lx,Ly, use_reentrant=False);
            
            Pos=convert_cell_posit(coord[1-1],coord[2-1],1,2,direction, Lx,Ly);
            AA=get_AA_direction(double_B_cell,double_T_cell,direction,Pos, ctm_setting, global_args);
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
            

            Cset_cell=update_cell(Cset_cell,Cset_new, Pos[1-1],Pos[2-1],Lx,Ly)



    return Cset_cell,Tset_cell


