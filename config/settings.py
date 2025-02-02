from collections import OrderedDict

def mod1(x,y):
    return 1+ (x-1)%y

def initial_cell(L1,L2):
    Cell=OrderedDict();
    for cx in range(1,L1+1):
        for cy in range(1,L2+1):
            Cell.update({str(cx)+','+str(cy):1.1})
    return Cell

def initial_Cset():
    Cell=OrderedDict();
    for cx in range(1,4+1):
        Cell.update({'C'+str(cx):1.1})
    return Cell
def initial_Tset():
    Cell=OrderedDict();
    for cx in range(1,4+1):
        Cell.update({'T'+str(cx):1.1})
    return Cell