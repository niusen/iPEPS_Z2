import torch
from collections import OrderedDict
import json
import math
import config as cfg
import yastn

class IPESS_TRIANGLE():
    def __init__(self, B_set, T_set,
                 peps_args=cfg.peps_args, global_args=cfg.global_args):


        self.B_set= B_set
        self.T_set= T_set
        sites = self.build_onsite_tensors()




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
        See :meth:`write_ipess_kagome_generic`.
        """
        write_ipess_kagome_generic(self, outputfile, tol=tol, normalize=normalize)

    def extend_bond_dim(self, new_d, peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param new_d: new enlarged auxiliary bond dimension
        :type state: IPESS_KAGOME_GENERIC
        :type new_d: int
        :return: wavefunction with enlarged auxiliary bond dimensions
        :rtype: IPESS_KAGOME_GENERIC

        Take IPESS_KAGOME_GENERIC and enlarge all auxiliary bond dimensions of ``T_u``, ``T_d``, 
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

def read_ipess_triangle(jsonfile, peps_args=cfg.peps_args, global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing iPEPS in JSON format`
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPESS_KAGOME_GENERIC

    Read state from file.
    """
    dtype = global_args.torch_dtype

    with open(jsonfile) as j:
        data = json.load(j)

    Bm_set=data['Bm_set']
    Tm_set=data['Tm_set']

    Lx=global_args.Lx;
    Ly=global_args.Ly;

    B_set=OrderedDict()
    T_set=OrderedDict()
    for cx in range(0,Lx):
        for cy in range(0,Ly):
            tm=Tm_set[str(cx+1)+','+str(cy+1)];
            bm=Bm_set[str(cx+1)+','+str(cy+1)];
            bm=convert_to_yastn(bm)
            tm=convert_to_yastn(tm)
            
            bm=yastn.Tensor.save_to_dict(bm)
            tm=yastn.Tensor.save_to_dict(tm)

            bm['_d']=(numpy.real(bm['_d'])).tolist()+(numpy.imag(bm['_d'])).tolist();
            tm['_d']=(numpy.real(bm['_d'])).tolist()+(numpy.imag(tm['_d'])).tolist();

            Bm_set.update({str(cx)+','+str(cy):bm})
            Tm_set.update({str(cx)+','+str(cy):tm})


    return state

def write_ipess_triangle(state, outputfile, tol=1.0e-14, normalize=False):
    r"""
    :param state: wavefunction to write out in json format
    :param outputfile: target file
    :param tol: minimum magnitude of tensor elements which are written out
    :param normalize: if True, on-site tensors are normalized before writing
    :type state: IPESS_KAGOME_GENERIC
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