import os
import sys
sys.path.append('D:/My Documents/Code/python_codes/iPEPS_Z2/test_yastn')


import json
import numpy
import torch
import yastn

# config_kwargs = {"backend": "np"}
config_kwargs = {"backend": "torch", "default_dtype": 'complex128'}


with open('SU_iPESS_Z2_csl_D4.JSON') as f:
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
        leg1 = yastn.Leg(config_Z2, s=2*(dual[0]-0.5), t=(0, 1), D=(even_dims[0], odd_dims[0]))
        leg2 = yastn.Leg(config_Z2, s=2*(dual[1]-0.5), t=(0, 1), D=(even_dims[1], odd_dims[1]))
        leg3 = yastn.Leg(config_Z2, s=2*(dual[2]-0.5), t=(0, 1), D=(even_dims[2], odd_dims[2]))
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
        leg1 = yastn.Leg(config_Z2, s=2*(dual[0]-0.5), t=(0, 1), D=(even_dims[0], odd_dims[0]))
        leg2 = yastn.Leg(config_Z2, s=2*(dual[1]-0.5), t=(0, 1), D=(even_dims[1], odd_dims[1]))
        leg3 = yastn.Leg(config_Z2, s=2*(dual[2]-0.5), t=(0, 1), D=(even_dims[2], odd_dims[2]))
        leg4 = yastn.Leg(config_Z2, s=2*(dual[3]-0.5), t=(0, 1), D=(even_dims[3], odd_dims[3]))
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









T_set=(data['T_set'])
B_set=(data['B_set'])
for cx in range(0,6) :
    for cy in range(0,6):
        tm=T_set[str(cx+1)+','+str(cy+1)];
        bm=B_set[str(cx+1)+','+str(cy+1)];
        bm=convert_to_yastn(bm)
        tm=convert_to_yastn(tm)

        if cx==3-1:
            if cy==5-1:
                bm=bm.to_dense()
                print(bm[1-1,3-1,3-1])
                
        # yastn.Tensor.save_to_dict(a) 
        # yastn.load_from_dict()

        

