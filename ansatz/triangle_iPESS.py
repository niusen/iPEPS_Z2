import numpy
import torch
import json
from collections import OrderedDict
import json
import yastn

class IPESS_TRIANGLE():
    def __init__(self, B_set, T_set, global_args):

        self.B_set= B_set;
        self.T_set= T_set;
        self.global_args=global_args;
        self.Lx=global_args['Lx']
        self.Ly=global_args['Ly']
        
    def require_grad(self,require):
        for cx in range(0,self.Lx):
            for cy in range(0,self.Ly):
                self.B_set[str(cx)+','+str(cy)].requires_grad_(requires_grad=require)
                self.T_set[str(cx)+','+str(cy)].requires_grad_(requires_grad=require)

    def to_device(self,device_):
        for cx in range(0,self.Lx):
            for cy in range(0,self.Ly):
                self.B_set[str(cx)+','+str(cy)]=self.B_set[str(cx)+','+str(cy)].to(device_)
                self.T_set[str(cx)+','+str(cy)]=self.T_set[str(cx)+','+str(cy)].to(device_)
        
    def normalize(self):
        for cx in range(0,self.Lx):
            for cy in range(0,self.Ly):
                self.B_set[str(cx)+','+str(cy)]=self.B_set[str(cx)+','+str(cy)]/(yastn.linalg.norm(self.B_set[str(cx)+','+str(cy)]))
                self.T_set[str(cx)+','+str(cy)]=self.T_set[str(cx)+','+str(cy)]/(yastn.linalg.norm(self.T_set[str(cx)+','+str(cy)]))

    def build_double_layer_iPESS(self):
        B_double_set=OrderedDict()
        T_double_set=OrderedDict()


def load_triangle_iPESS(filenm,config_kwargs):
    with open(filenm+'.JSON') as f:
        data = json.load(f)
        # print(data)

    def convert_to_yastn(Dict):
        T_real=numpy.array(Dict['T_real'], order='F');
        T_imag=numpy.array(Dict['T_imag'], order='F');
        T=T_real+1j*T_imag;
        assert len(numpy.shape(T))==1
        
        even_dims=numpy.array(Dict['even_dims']);
        odd_dims=numpy.array(Dict['odd_dims']);
        dims=even_dims+odd_dims;
        if len(dims)==3:
            T=numpy.reshape(T,(dims[0],dims[1],dims[2]),order='F')
        elif len(dims)==4:
            T=numpy.reshape(T,(dims[0],dims[1],dims[2],dims[3]),order='F')
        
        dual=Dict['dual'];
        if len(dual)==3:
            config_Z2 = yastn.make_config(sym='Z2',fermionic=True, **config_kwargs)
            leg1 = yastn.Leg(config_Z2, s=2*(0.5-dual[0]), t=(0, 1), D=(even_dims[0], odd_dims[0]))
            leg2 = yastn.Leg(config_Z2, s=2*(0.5-dual[1]), t=(0, 1), D=(even_dims[1], odd_dims[1]))
            leg3 = yastn.Leg(config_Z2, s=2*(0.5-dual[2]), t=(0, 1), D=(even_dims[2], odd_dims[2]))
            tt = yastn.zeros(config=config_Z2, legs=[leg1, leg2, leg3])

            for c1 in range(0,2):
                if c1==0:
                    range1=range(0,even_dims[0])
                else:
                    range1=range(even_dims[0],even_dims[0]+odd_dims[0])
                for c2 in range(0,2):
                    if c2==0:
                        range2=range(0,even_dims[1])
                    else:
                        range2=range(even_dims[1],even_dims[1]+odd_dims[1])
                    for c3 in range(0,2):
                        if c3==0:
                            range3=range(0,even_dims[2])
                        else:
                            range3=range(even_dims[2],even_dims[2]+odd_dims[2])

                        T_=T[range1,:,:]
                        T_=T_[:,range2,:]
                        T_=T_[:,:,range3]

                        if numpy.mod(c1+c2+c3,2)==0:
                            tt.set_block(ts=(c1, c2, c3), val=T_, Ds=(len(range1), len(range2), len(range3)))
        elif len(dual)==4:
            config_Z2 = yastn.make_config(sym='Z2',fermionic=True, **config_kwargs)
            leg1 = yastn.Leg(config_Z2, s=2*(0.5-dual[0]), t=(0, 1), D=(even_dims[0], odd_dims[0]))
            leg2 = yastn.Leg(config_Z2, s=2*(0.5-dual[1]), t=(0, 1), D=(even_dims[1], odd_dims[1]))
            leg3 = yastn.Leg(config_Z2, s=2*(0.5-dual[2]), t=(0, 1), D=(even_dims[2], odd_dims[2]))
            leg4 = yastn.Leg(config_Z2, s=2*(0.5-dual[3]), t=(0, 1), D=(even_dims[3], odd_dims[3]))
            tt = yastn.zeros(config=config_Z2, legs=[leg1, leg2, leg3, leg4])

            for c1 in range(0,2):
                if c1==0:
                    range1=range(0,even_dims[0])
                else:
                    range1=range(even_dims[0],even_dims[0]+odd_dims[0])
                for c2 in range(0,2):
                    if c2==0:
                        range2=range(0,even_dims[1])
                    else:
                        range2=range(even_dims[1],even_dims[1]+odd_dims[1])
                    for c3 in range(0,2):
                        if c3==0:
                            range3=range(0,even_dims[2])
                        else:
                            range3=range(even_dims[2],even_dims[2]+odd_dims[2])
                        for c4 in range(0,2):
                            if c4==0:
                                range4=range(0,even_dims[3])
                            else:
                                range4=range(even_dims[3],even_dims[3]+odd_dims[3])

                            T_=T[range1,:,:,:]
                            T_=T_[:,range2,:,:]
                            T_=T_[:,:,range3,:]
                            T_=T_[:,:,:,range4]

                            if numpy.mod(c1+c2+c3+c4,2)==0:
                                tt.set_block(ts=(c1, c2, c3, c4), val=T_, Ds=(len(range1), len(range2), len(range3), len(range4)))
        return tt

    Lx=config_kwargs['Lx'];
    Ly=config_kwargs['Ly'];
    T_set=(data['T_set'])
    B_set=(data['B_set'])
    Bm_set=OrderedDict()
    Tm_set=OrderedDict()
    for cx in range(0,Lx):
        for cy in range(0,Ly):
            tm=T_set[str(cx+1)+','+str(cy+1)];
            bm=B_set[str(cx+1)+','+str(cy+1)];
            bm=convert_to_yastn(bm)
            tm=convert_to_yastn(tm)
            
            # bm=yastn.Tensor.save_to_dict(bm)
            # tm=yastn.Tensor.save_to_dict(tm)
            # bm['_d']=(numpy.real(bm['_d'])).tolist()+(numpy.imag(bm['_d'])).tolist();
            # tm['_d']=(numpy.real(bm['_d'])).tolist()+(numpy.imag(tm['_d'])).tolist();

            Bm_set.update({str(cx)+','+str(cy):bm})
            Tm_set.update({str(cx)+','+str(cy):tm})

            # yastn.load_from_dict()
    return Bm_set,Tm_set

def save_triangle_iPESS(Bm_set, Tm_set, filenm, config_kwargs):
    def yastn_to_dict(TT):
        sigs=TT.s;
        dual=(0.5-numpy.array(sigs)/2).astype(int).tolist();
        Rank=len(sigs);
        legs=TT.get_legs();
    
        if Rank==3:
            even_dims=[legs[0].D[0], legs[1].D[0], legs[2].D[0]]
            odd_dims=[legs[0].D[1], legs[1].D[1], legs[2].D[1]]
        elif Rank==4:
            even_dims=[legs[0].D[0], legs[1].D[0], legs[2].D[0], legs[3].D[0]]
            odd_dims=[legs[0].D[1], legs[1].D[1], legs[2].D[1], legs[3].D[1]]
        TT_dense=numpy.array(TT.to_dense().to('cpu'))
        TT_dense=numpy.reshape(TT_dense,-1,order='F');
        T_real=numpy.real(TT_dense);
        T_imag=numpy.imag(TT_dense);
        Dict={'T_real':T_real.tolist(),'T_imag':T_imag.tolist(),'even_dims':even_dims,'odd_dims':odd_dims,'dual':dual}
        return Dict

    B_set=OrderedDict()
    T_set=OrderedDict()
    Lx=config_kwargs['Lx'];
    Ly=config_kwargs['Ly'];

    for cx in range(0,Lx):
        for cy in range(0,Ly):
            bm=Bm_set[str(cx)+','+str(cy)]
            tm=Tm_set[str(cx)+','+str(cy)]
            bm=yastn_to_dict(bm)
            tm=yastn_to_dict(tm)
            
            B_set.update({str(cx+1)+','+str(cy+1):bm})
            T_set.update({str(cx+1)+','+str(cy+1):tm})

    with open(filenm+'.json', "w") as f:
        json.dump({'T_set':T_set,'B_set':B_set}, f)

 
# # config_kwargs = {"backend": "np"}
# config_kwargs = {"backend": "torch", "default_dtype": 'complex128', 'default_device': 'cuda', 'Lx':6, 'Ly':6}

# filenm='SU_iPESS_Z2_csl_D4'
# Bm_set,Tm_set=load_triangle_iPESS(filenm,config_kwargs)

# filenm1='SU_iPESS_Z2_D4'
# save_triangle_iPESS(Bm_set, Tm_set, filenm1, config_kwargs)


# Bm_set2,Tm_set2=load_triangle_iPESS(filenm1,config_kwargs)

# #verify save and reload
# for cx in range(0,6):
#     for cy in range(0,6):
#         print(yastn.linalg.norm(Bm_set[str(cx)+','+str(cy)]-Bm_set2[str(cx)+','+str(cy)]));
#         print(yastn.linalg.norm(Tm_set[str(cx)+','+str(cy)]-Tm_set2[str(cx)+','+str(cy)]));