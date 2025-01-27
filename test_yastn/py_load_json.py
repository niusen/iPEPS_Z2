import json
import numpy


with open('write_read.json') as f:
    data = json.load(f)
    print(data)



T_real=numpy.array((data['T_real']))
T_imag=numpy.array((data['T_imag']))