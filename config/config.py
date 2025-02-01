import torch
import argparse
import logging



class GLOBALARGS():

    def __init__(self):
        self.tensor_io_format= "legacy"
        self.dtype= "float64"
        self.torch_dtype= torch.float64
        self.device= 'cpu'
        self.offload_to_gpu= 'None'
        self.Lx=6;
        self.Ly=6;

    def __str__(self):
        res=type(self).__name__+"\n"
        for x in list(filter(lambda x: "__" not in x,dir(self))):
            res+=f"{x}= {getattr(self,x)}\n"
        return res[:-1]


class CTMARGS():
    def __init__(self):
        self.ctm_max_iter= 50
        self.ctm_env_init_type= 'CTMRG'
        self.ctm_conv_tol= 1.0e-8
        self.ctm_absorb_normalization= 'inf'
        self.fpcm_init_iter=1
        self.fpcm_freq= -1
        self.fpcm_isogauge_tol= 1.0e-14
        self.fpcm_fpt_tol= 1.0e-8
        self.conv_check_cpu = False
        self.projector_method = '4X4'
        self.projector_svd_method = 'DEFAULT'
        self.projector_svd_reltol = 1.0e-8
        self.projector_svd_reltol_block = 0.0
        self.projector_eps_multiplet = 1.0e-8
        self.projector_multiplet_abstol = 1.0e-14
        self.ad_decomp_reg= 1.0e-12
        self.ctm_move_sequence = [(0,-1), (-1,0), (0,1), (1,0)]
        self.randomize_ctm_move_sequence = False
        self.ctm_force_dl = False
        self.ctm_logging = False
        self.verbosity_initialization = 0
        self.verbosity_ctm_convergence = 0
        self.verbosity_projectors = 0
        self.verbosity_ctm_move = 0
        self.verbosity_fpcm_move = 0
        self.verbosity_rdm = 0
        self.fwd_checkpoint_c2x2 = False
        self.fwd_checkpoint_halves = False
        self.fwd_checkpoint_projectors = False
        self.fwd_checkpoint_absorb = False
        self.fwd_checkpoint_move = False
        self.fwd_checkpoint_loop_rdm = False

    def __str__(self):
        res=type(self).__name__+"\n"
        for x in list(filter(lambda x: "__" not in x,dir(self))):
            res+=f"{x}= {getattr(self,x)}\n"
        return res[:-1]

class OPTARGS():

    def __init__(self):
        self.lr= 1.0
        self.momentum= 0.
        self.dampening= 0.
        self.tolerance_grad= 1e-5
        self.tolerance_change= 1e-9
        self.opt_ctm_reinit= True
        self.env_sens_scale= 10.0
        self.line_search= "default"
        self.line_search_ctm_reinit= True
        self.line_search_svd_method= 'DEFAULT'
        self.line_search_tol= 1.0e-8
        self.fd_eps= 1.0e-4
        self.fd_ctm_reinit= True
        self.history_size= 100
        self.max_iter_per_epoch= 1
        self.verbosity_opt_epoch= 1
        self.opt_logging= True
        self.opt_log_grad= False

    def __str__(self):
        res=type(self).__name__+"\n"
        for x in list(filter(lambda x: "__" not in x,dir(self))):
            res+=f"{x}= {getattr(self,x)}\n"
        return res[:-1]


global_args= GLOBALARGS()
ctm_args= CTMARGS()
opt_args= OPTARGS()
