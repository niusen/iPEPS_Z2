import torch
import argparse
import logging

class MAINARGS():
    r"""
    Main simulation options. The default settings can be modified through 
    command line arguments as follows ``--<option-name> desired-value``

    :ivar omp_cores: number of OpenMP cores. Default: ``1``
    :vartype omp_cores: int:
    :ivar instate: input state file. Default: ``None``
    :vartype instate: str or Path
    :ivar instate_noise: magnitude of noise applied to the input state, if any. Default: ``0.0``
    :vartype instate_noise: float
    :ivar ipeps_init_type: initialization of the trial iPEPS state, if no ``instate`` is provided. Default: ``RANDOM``
    :vartype ipeps_init_type: str
    :ivar out_prefix: output file prefix. Default: ``output``
    :vartype out_prefix: str
    :ivar bond_dim: iPEPS auxiliary bond dimension. Default: ``1``
    :vartype bond_dim: int
    :ivar chi: environment bond dimension. Default: ``20``
    :vartype chi: int
    :ivar opt_max_iter: maximal number of optimization steps. Default: ``100``
    :vartype opt_max_iter: int
    :ivar opt_resume: resume from checkpoint file. Default: ``None``
    :vartype opt_resume: str or Path
    :ivar opt_resume_override_params: override optimizer parameters stored in checkpoint. Default: ``False``
    :vartype opt_resume_override_params: bool
    :ivar seed: PRNG seed. Default: ``0``
    :vartype seed: int
    """
    def __init__(self):
        self.opt_max_iter= 100
        self.out_prefix=None
        self.opt_resume=None
        self.opt_resume_override_params=None

    def __str__(self):
        res=type(self).__name__+"\n"
        for x in list(filter(lambda x: "__" not in x,dir(self))):
            res+=f"{x}= {getattr(self,x)}\n"
        return res[:-1]

class LINESEARCH():

    def __init__(self):
        self.maxiter=100
        self.gtol=1e-3
        self.delta0=1e-3
        self.alpha= 3/4

    def __str__(self):
        res=type(self).__name__+"\n"
        for x in list(filter(lambda x: "__" not in x,dir(self))):
            res+=f"{x}= {getattr(self,x)}\n"
        return res[:-1]

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
    
class INITCTMARGS():

    def __init__(self):
        self.reconstruct_AA= True
        self.reconstruct_CTM= True

    def __str__(self):
        res=type(self).__name__+"\n"
        for x in list(filter(lambda x: "__" not in x,dir(self))):
            res+=f"{x}= {getattr(self,x)}\n"
        return res[:-1]
    
class CTMARGS():
    def __init__(self):
        self.chi= 10
        self.CTM_ite_nums= 50
        self.CTM_conv_tol= 1.0e-6
        self.projector_strategy = '4x4'
        self.CTM_trun_svd = False
        self.CTM_trun_tol = 1.0e-10
        self.svd_lanczos_tol = 1e-10
        self.CTM_ite_info = False
        self.construct_double_layer=True;
        self.use_reentrant=False;
        self.use_checkpoint=True;
        self.checkpoint_device='cpu';
    
        #There are currently two checkpointing implementations available, determined
        #by the :attr:`use_reentrant` parameter. It is recommended that you use
        #``use_reentrant=False``. Please refer the note below for a discussion of
        #their differences.

        #The reentrant version does not consider tensors in nested structures
        #(e.g., custom objects, lists, dicts, etc) as participating in
        #autograd, while the non-reentrant version does.


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


class Square_Hubbard_Energy_settings():

    def __init__(self):
        self.model= 'spinful_triangle_lattice'

    def __str__(self):
        res=type(self).__name__+"\n"
        for x in list(filter(lambda x: "__" not in x,dir(self))):
            res+=f"{x}= {getattr(self,x)}\n"
        return res[:-1]