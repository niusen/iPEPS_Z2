from collections import OrderedDict

def mod1(x,y):
    return 1+ (x-1)%y
def mod(x,y):
    return (x)%y

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

def update_CTM_C(Cset0,TT,Ind):
    Cset=OrderedDict();
    for cx in range(1,4+1):
        if cx==Ind:
            Cset.update({'C'+str(cx):TT})
        else:
            Cset.update({'C'+str(cx):Cset0['C'+str(cx)]})
    return Cset
def update_CTM_T(Tset0,TT,Ind):
    Tset=OrderedDict();
    for cx in range(1,4+1):
        if cx==Ind:
            Tset.update({'T'+str(cx):TT})
        else:
            Tset.update({'T'+str(cx):Tset0['T'+str(cx)]})
    return Tset

def update_cell(Cell_old,ele,indx,indy,Lx,Ly):
    Cell=OrderedDict();
    for cx in range(1,Lx+1):
        for cy in range(1,Ly+1):
            if (indx==cx)&(indy==cy):
                Cell[str(cx)+','+str(cy)]=ele;
            else:
                Cell[str(cx)+','+str(cy)]=Cell_old[str(cx)+','+str(cy)];
    return Cell
