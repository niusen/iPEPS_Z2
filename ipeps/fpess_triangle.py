import torch
from collections import OrderedDict
import json
import math
import config as cfg
from ipeps.ipeps_triangle import IPEPS_TRIANGLE
from ipeps.tensor_io import *

class IPESS_TRIANGLE_GENERIC(IPEPS_TRIANGLE):
    def __init__(self, ipess_tensors,
                 peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param ipess_tensors: dictionary of five tensors, which make up Triangle iPESS
                              ansatz 
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type ipess_tensors: dict(str, torch.tensor)
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        iPESS ansatz for Triangle lattice composes five tensors, specified
        by dictionary::

            ipess_tensors = {'T_u': torch.Tensor, 'T_d': ..., 'B_a': ..., 'B_b': ..., 'B_c': ...}

        into a single rank-5 on-site tensor of parent IPEPS. These iPESS tensors can
        be accessed through member ``ipess_tensors``. 

        The ``'B_*'`` are rank-3 tensors, with index structure [p,i,j] where the first index `p` 
        is for physical degree of freedom, while indices `i` and `j` are auxiliary with bond dimension D.
        These bond tensors reside on corners shared between different triangles of Triangle lattice. 
        Bond tensors are connected by rank-3 trivalent tensors ``'T_u'``, ``'T_d'`` on up and down triangles 
        respectively. Trivalent tensors have only auxiliary indices of matching bond dimension D.
    
        The on-site tensors of corresponding iPEPS is obtained by the following contraction::
                                                     
                 2(d)            2(c)                    a
                  \             /          rot. pi       |
             0(w)==B_a         B_b==0(v)   clockwise  b--\                     
                    \         /             =>            \
                    1(l)     1(k)                         s0--s2--d
                     2(l)   1(k)                           | / 
                       \   /                               |/   <- down triangle
                        T_d                               s1
                         |                                 |
                         0(j)                              c
                         1(j)                               
                         |                 
                         B_c==0(u)        
                         |
                         2(i)
                         0(i)  
                         |
                        T_u
                       /   \ 
                     1(a)   2(b) 

        By construction, the degrees of freedom on down triangle are all combined into 
        a single on-site tensor of iPEPS. Instead, DoFs on the upper triangle have 
        to be accessed by construction of 2x2 patch (which is then embedded into environment)::        
        
            C    T             T          C
                 a             a
                 |             |
            T b--\          b--\
                  \        /    \
                  s0--s2--d     s0--s2--d T
                   | /           | /
                   |/            |/
                  s1            s1
                   |             |
                   c             c  
                  /             /
                 a             a
                 |             |
            T b--\          b--\
                  \        /    \
                  s0--s2--d     s0--s2--d T
                   | /           | /
                   |/            |/
                  s1            s1
                   |             |
                   c             c
            C      T             T        C
        """
        #TODO verification?
        self.ipess_tensors= ipess_tensors
        sites = self.build_onsite_tensors()

        super().__init__(sites, lX=1, lY=1, peps_args=peps_args,
                         global_args=global_args)

    def get_parameters(self):
        r"""
        :return: variational parameters of IPESS_TRIANGLE_GENERIC
        :rtype: iterable
        
        This function is called by optimizer to access variational parameters of the state.
        In this case member ``ipess_tensors``.
        """
        return self.ipess_tensors.values()

    def get_checkpoint(self):
        r"""
        :return: all data necessary to reconstruct the state. In this case member ``ipess_tensors`` 
        :rtype: dict[str: torch.tensor]
        
        This function is called by optimizer to create checkpoints during 
        the optimization process.
        """
        return self.ipess_tensors

    def load_checkpoint(self, checkpoint_file):
        checkpoint= torch.load(checkpoint_file, map_location=self.device)
        self.ipess_tensors= checkpoint["parameters"]
        for t in self.ipess_tensors.values(): t.requires_grad_(False)
        self.sites = self.build_onsite_tensors()

    def build_onsite_tensors(self):
        r"""
        :return: elementary unit cell of underlying IPEPS
        :rtype: dict[tuple(int,int): torch.Tensor]

        Build rank-5 on-site tensor by contracting the iPESS tensors.
        """
        A= torch.einsum('iab,uji,jkl,vkc,wld->uvwabcd', self.ipess_tensors['T_u'],
            self.ipess_tensors['B_c'], self.ipess_tensors['T_d'], self.ipess_tensors['B_b'], \
            self.ipess_tensors['B_a'])
        total_phys_dim= self.ipess_tensors['B_a'].size(0)*self.ipess_tensors['B_b'].size(0)\
            *self.ipess_tensors['B_c'].size(0)
        A= A.reshape([total_phys_dim]+[self.ipess_tensors['T_u'].size(1), \
            self.ipess_tensors['T_u'].size(2), self.ipess_tensors['B_b'].size(2), \
            self.ipess_tensors['B_a'].size(2)])
        A= A/A.abs().max()
        sites= {(0, 0): A}
        return sites

    def add_noise(self, noise):
        r"""
        :param noise: magnitude of noise
        :type noise: float

        Add uniform random noise to iPESS tensors.
        """
        for k in self.ipess_tensors:
            rand_t= torch.rand( self.ipess_tensors[k].size(), dtype=self.dtype, device=self.device)
            self.ipess_tensors[k]= self.ipess_tensors[k] + noise * (rand_t-1.0)
        self.sites = self.build_onsite_tensors()

    def get_physical_dim(self):
        assert self.ipess_tensors["B_a"].size(0)==self.ipess_tensors["B_b"].size(0) and \
            self.ipess_tensors["B_b"].size(0)==self.ipess_tensors["B_c"].size(0),\
            "Different physical dimensions across iPESS bond tensors"
        return self.ipess_tensors["B_a"].size(0)

    def get_aux_bond_dims(self):
        aux_bond_dims= set()
        aux_bond_dims= aux_bond_dims | set(self.ipess_tensors["T_u"].size()) \
            | set(self.ipess_tensors["T_d"].size())
        assert len(aux_bond_dims)==1,"iPESS does not have a uniform aux bond dimension"
        return list(aux_bond_dims)[0]

    def write_to_file(self, outputfile, aux_seq=None, tol=1.0e-14, normalize=False):
        r"""
        See :meth:`write_ipess_triangle_generic`.
        """
        write_ipess_triangle_generic(self, outputfile, tol=tol, normalize=normalize)

    def extend_bond_dim(self, new_d, peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param new_d: new enlarged auxiliary bond dimension
        :type state: IPESS_TRIANGLE_GENERIC
        :type new_d: int
        :return: wavefunction with enlarged auxiliary bond dimensions
        :rtype: IPESS_TRIANGLE_GENERIC

        Take IPESS_TRIANGLE_GENERIC and enlarge all auxiliary bond dimensions of ``T_u``, ``T_d``, 
        ``B_a``, ``B_b``, and ``B_c`` tensors to the new size ``new_d``.
        """
        ad= self.get_aux_bond_dims()
        assert new_d>=ad, "Desired dimension is smaller than current aux dimension"
        new_ipess_tensors= dict()
        for k in ['T_u','T_d']:
            new_ipess_tensors[k]= torch.zeros(new_d,new_d,new_d, dtype=self.dtype, device=self.device)
            new_ipess_tensors[k][:ad,:ad,:ad]= self.ipess_tensors[k]
        for k in ['B_a','B_b', 'B_c']:
            new_ipess_tensors[k]= torch.zeros(self.ipess_tensors[k].size(0),new_d,new_d,\
                dtype=self.dtype, device=self.device)
            new_ipess_tensors[k][:,:ad,:ad]= self.ipess_tensors[k]

        new_state= self.__class__(new_ipess_tensors,\
            peps_args=peps_args, global_args=global_args)

        return new_state
def read_ipess_triangle_from_julia(jsonfile, peps_args=cfg.peps_args, global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing iPEPS in JSON format`
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPESS_TRIANGLE_GENERIC

    Read state from file.
    """
    dtype = global_args.torch_dtype

    with open(jsonfile) as j:
        raw_state = json.load(j)

        # Loop over non-equivalent tensor,coeffs pairs in the unit cell
        ipess_tensors= OrderedDict()
        # legacy
        if "elem_tensors" in raw_state.keys():
            assert set(("UP_T","DOWN_T","BOND_S1","BOND_S2","BOND_S3"))\
                ==set(list(raw_state["elem_tensors"].keys())),"missing elementary tensors"
            keymap={"UP_T": "T_u", "DOWN_T": "T_d", "BOND_S1": "B_c","BOND_S3": "B_a","BOND_S2": "B_b"}
            for key,t in raw_state["elem_tensors"].items():
                ipess_tensors[keymap[key]]= torch.from_numpy(read_bare_json_tensor_np_legacy(t))

        # default
        elif "ipess_tensors" in raw_state.keys(): 
            assert set(('T_u','T_d','B_a','B_b','B_c'))==set(list(raw_state["ipess_tensors"].keys())),\
                "missing ipess tensors"
            for key,t in raw_state["ipess_tensors"].items():
                ipess_tensors[key]= torch.from_numpy(read_bare_json_tensor_np_legacy(t))
        else:
            raise RuntimeError("Not a valid IPESS_TRIANGLE_GENERIC state.")

        # convert to correct device and/or dtype
        for key,t in ipess_tensors.items():
             ipess_tensors[key]=  ipess_tensors[key].to(device=global_args.device,dtype=dtype)                  

        state = IPESS_TRIANGLE_GENERIC(ipess_tensors, peps_args=peps_args, \
            global_args=global_args)
    return state
def read_ipess_triangle_generic(jsonfile, peps_args=cfg.peps_args, global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing iPEPS in JSON format`
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPESS_TRIANGLE_GENERIC

    Read state from file.
    """
    dtype = global_args.torch_dtype

    with open(jsonfile) as j:
        raw_state = json.load(j)

        # Loop over non-equivalent tensor,coeffs pairs in the unit cell
        ipess_tensors= OrderedDict()
        # legacy
        if "elem_tensors" in raw_state.keys():
            assert set(("UP_T","DOWN_T","BOND_S1","BOND_S2","BOND_S3"))\
                ==set(list(raw_state["elem_tensors"].keys())),"missing elementary tensors"
            keymap={"UP_T": "T_u", "DOWN_T": "T_d", "BOND_S1": "B_c","BOND_S3": "B_a","BOND_S2": "B_b"}
            for key,t in raw_state["elem_tensors"].items():
                ipess_tensors[keymap[key]]= torch.from_numpy(read_bare_json_tensor_np_legacy(t))

        # default
        elif "ipess_tensors" in raw_state.keys(): 
            assert set(('T_u','T_d','B_a','B_b','B_c'))==set(list(raw_state["ipess_tensors"].keys())),\
                "missing ipess tensors"
            for key,t in raw_state["ipess_tensors"].items():
                ipess_tensors[key]= torch.from_numpy(read_bare_json_tensor_np_legacy(t))
        else:
            raise RuntimeError("Not a valid IPESS_TRIANGLE_GENERIC state.")

        # convert to correct device and/or dtype
        for key,t in ipess_tensors.items():
             ipess_tensors[key]=  ipess_tensors[key].to(device=global_args.device,dtype=dtype)                  

        state = IPESS_TRIANGLE_GENERIC(ipess_tensors, peps_args=peps_args, \
            global_args=global_args)
    return state

def write_ipess_triangle_generic(state, outputfile, tol=1.0e-14, normalize=False):
    r"""
    :param state: wavefunction to write out in json format
    :param outputfile: target file
    :param tol: minimum magnitude of tensor elements which are written out
    :param normalize: if True, on-site tensors are normalized before writing
    :type state: IPESS_TRIANGLE_GENERIC
    :type ouputfile: str or Path object
    :type tol: float
    :type normalize: bool

    Write state into file.
    """
    #TODO implement cutoff on elements with magnitude below tol
    json_state = dict({"lX": state.lX, "lY": state.lY, \
        "ipess_tensors": {}})

    # write list of considered elementary tensors
    for key, t in state.ipess_tensors.items():
        tmp_t= t/t.abs().max() if normalize else t
        json_state["ipess_tensors"][key]= serialize_bare_tensor_legacy(tmp_t)

    with open(outputfile, 'w') as f:
        json.dump(json_state, f, indent=4, separators=(',', ': '))

