import os
import sys
sys.path.append('D:/My Documents/Code/python_codes/iPEPS_Z2')
from collections import OrderedDict
import json
import numpy
import torch
import yastn
from timeit import default_timer as timer


# config_kwargs = {"backend": "np"}
#device: 'cpu', 'cuda'
config_kwargs = {"backend": "torch", "default_dtype": 'complex128', 'default_device': 'cuda', 'Lx':6, 'Ly':6}
config_Z2 = yastn.make_config(sym='Z2', **config_kwargs)
print(config_Z2.backend.cuda_is_available())
D0=1000;
leg1 = yastn.Leg(config_Z2, s=1, t=( 0, 1), D=(D0, D0))
leg2 = yastn.Leg(config_Z2, s=-1, t=( 0, 1), D=(D0, D0))

T = yastn.rand(config=config_Z2, legs=[leg1, leg2])
print(T)
print(T.dtype)



T=T.to('cpu');
print(T.device)
start = timer()
TT = yastn.ncon([T, T], [[-1, 1], [1, -2]])
end = timer()
print(end - start) 




T=T.to(device='cuda:0');
print(T.device)
start = timer()
TT = yastn.ncon([T, T], [[-1, 1], [1, -2]])
end = timer()
print(end - start) 

print(T.device)
start = timer()
TT = yastn.ncon([T, T], [[-1, 1], [1, -2]])
end = timer()
print(end - start) 


T=T.to('cpu');
print(T.device)
start = timer()
U,S,V=yastn.linalg.svd(T)
end = timer()
print(end - start) 

T=T.to(device='cuda:0');
print(T.device)
start = timer()
U,S,V=yastn.linalg.svd(T)
end = timer()
print(end - start) 

T=T.to(device='cuda:0');
print(T.device)
start = timer()
U,S,V=yastn.linalg.svd(T,svd_on_cpu=True)
end = timer()
print(end - start) 




        

