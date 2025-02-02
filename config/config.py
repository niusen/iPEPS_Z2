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


