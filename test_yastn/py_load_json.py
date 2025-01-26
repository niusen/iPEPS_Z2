import json
import numpy


with open('write_read.json') as f:
    d = json.load(f)
    print(d)



T_real=numpy.array((d['T_real']))
T_imag=numpy.array((d['T_imag']))