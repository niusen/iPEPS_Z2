"""Complex physical-NN SU; reuse old QR/SVD algorithm and orbit conventions.

No symmetry/magnetization projection. All 54 physical bonds and all 27 field
terms occur once per Strang step, including when J=0. Lambda weights stay real.
"""
from functools import lru_cache
import torch
from ...simple_update import apply_local_factors,copy_rotation_orbit
from ...geometry import neighbour,edge_bonds
from ...paper_method.physical_nn_su import apply_inside,field_diagonal,orbit_representatives
from .linalg import full_svd
from .model import physical_gate,arrow_sign,complex_state


@lru_cache(None)
def cross_factors(dt,J,Dz,sign):
    g=physical_gate(dt,J,Dz,sign).reshape(2,2,2,2).permute(0,2,1,3).reshape(4,4)
    u,s,vh=full_svd(g);keep=s>1e-14*s[0]
    return ((u[:,keep]*s[keep].sqrt()).T.reshape(-1,2,2),
            (vh[keep]*s[keep].sqrt()[:,None]).reshape(-1,2,2))


@torch.no_grad()
def update_cross(state,c,k,i,j,dt,J,Dz):
    if (i,j) not in edge_bonds(k):raise ValueError('Not a physical NN bond')
    b=neighbour(c,k);D=state.D
    order=[0]+[l+1 for l in range(3) if l!=k]+[k+1];inverse=[order.index(l) for l in range(4)]
    a,t=state.A[c].clone(),state.B[b].clone()
    for kind,colour,x in (('A',c,a),('B',b,t)):
        for leg,w in enumerate(state.lambdas(kind,colour)):
            if leg!=k:
                shape=[1]*4;shape[leg+1]=D;x*=w.to(x.device).reshape(shape)
    ga,gb=cross_factors(float(dt),float(J),float(Dz),arrow_sign('cross',k,i,j))
    x=apply_local_factors(a.permute(order),ga.to(a),[i],6)
    y=apply_local_factors(t.permute(order),gb.to(t),[j],3)
    w=state.weights[c][k].to(a.device).sqrt().reshape(1,1,1,D,1)
    x=(x*w).reshape(64*D*D,-1);y=(y*w).reshape(8*D*D,-1)
    qx,rx=torch.linalg.qr(x,mode='reduced');qy,ry=torch.linalg.qr(y,mode='reduced')
    # These are TWO KET tensors: transpose, not Hermitian conjugate.
    u,s,vh=full_svd(rx@ry.T)
    discard=float(s[D:].square().sum()/s.square().sum())
    a=(qx@u[:,:D]).reshape(64,D,D,D).permute(inverse)
    t=(qy@vh[:D].T).reshape(8,D,D,D).permute(inverse)
    new_w=s[:D]/s[:D].norm();change=float((new_w-state.weights[c][k].to(new_w)).abs().max())
    for kind,colour,x in (('A',c,a),('B',b,t)):
        for leg,w in enumerate(state.lambdas(kind,colour)):
            if leg!=k:
                shape=[1]*4;shape[leg+1]=D;x/=w.to(x.device).clamp_min(1e-12).reshape(shape)
    state.A[c],state.B[b]=a/a.norm(),t/t.norm();state.weights[c][k]=new_w
    return change,discard


@torch.no_grad()
def update_orbit(state,term,dt,J,Dz):
    result=[]
    for r in ((0,) if state.c3 else range(3)):
        if term[0]=='field':
            kind=term[1];group=state.A if kind=='A' else state.B;c=r if kind=='A' else (-r)%3
            group[c]=group[c]*field_diagonal(kind,state.field,float(dt)).to(group[c]).reshape(-1,1,1,1)
            group[c]=group[c]/group[c].norm()
            if state.c3:copy_rotation_orbit(state,kind,c)
        elif term[0]=='internal':
            i=(term[1]-2*r)%6;j=(i+1)%6
            sign=arrow_sign('internal',-1,i,j)
            state.A[r]=apply_inside(state.A[r],physical_gate(float(dt),J,Dz,sign),[i,j],6)
            if state.c3:copy_rotation_orbit(state,'A',r)
        else:
            _,k,i,j=term;k=(k-r)%3;i=(i-2*r)%6;j=(j-r)%3
            result.append(update_cross(state,r,k,i,j,dt,J,Dz))
            if state.c3:
                copy_rotation_orbit(state,'A',0);copy_rotation_orbit(state,'B',neighbour(0,k))
                for a in (1,2):state.weights[a][(k-a)%3]=state.weights[0][k].clone()
    return max((a[0] for a in result),default=0.),max((a[1] for a in result),default=0.)


@torch.no_grad()
def sweep(state,dt,*,J=1.,Dz=0.,second_order=True):
    complex_state(state)
    catalog=orbit_representatives();sequence=catalog+catalog[::-1] if second_order else catalog
    values=[update_orbit(state,t,dt/2 if second_order else dt,float(J),float(Dz)) for t in sequence]
    state.steps+=1
    return max(a[0] for a in values),max(a[1] for a in values)
