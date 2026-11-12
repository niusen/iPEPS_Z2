"""SU with an SVD after EACH inter-block physical nearest-neighbour gate.

The paper specifies NN two-site gates but does not disclose their exact
blocked implementation. This is an explicit interpretation of that recipe,
not a claim that the author's unpublished sequence has been recovered.
Intra-hexagon and onsite gates act exactly inside their physical block.
All 54 bonds and all 27 field terms occur once per full Strang step.
"""
from functools import lru_cache
import torch
from ..simple_update import (SZ,SP,SM,ID,apply_local_factors,
                             copy_rotation_orbit)
from ..geometry import edge_bonds,neighbour


@lru_cache(None)
def physical_gate(dt):
    h=torch.kron(SZ,SZ)+.5*(torch.kron(SP,SM)+torch.kron(SM,SP))
    return torch.matrix_exp(-float(dt)*h)


@lru_cache(None)
def cross_factors(dt):
    g=physical_gate(dt).reshape(2,2,2,2).permute(0,2,1,3).reshape(4,4)
    u,s,vh=torch.linalg.svd(g,full_matrices=False)
    keep=s>1e-14*s[0]
    return ((u[:,keep]*s[keep].sqrt()).T.reshape(-1,2,2),
            (vh[keep]*s[keep].sqrt()[:,None]).reshape(-1,2,2))


def apply_inside(t,gate,sites,n):
    inactive=[i for i in range(n) if i not in sites]
    order=list(sites)+inactive+list(range(n,t.ndim+n-1))
    shape=([2]*n)+list(t.shape[1:])
    x=t.reshape(shape).permute(order)
    x=(gate.to(t)@x.reshape(2**len(sites),-1)).reshape([shape[i] for i in order])
    inverse=[order.index(i) for i in range(len(order))]
    x=x.permute(inverse).reshape_as(t)
    return x/x.norm()


@torch.no_grad()
def update_cross(state,c,k,i,j,dt):
    """Physical A_i--B_j gate, with the standard diagonal-lambda SU metric."""
    if (i,j) not in edge_bonds(k):
        raise ValueError('Gate is not a physical kagome NN bond')
    b=neighbour(c,k);D=state.D
    order=[0]+[l+1 for l in range(3) if l!=k]+[k+1]
    inverse=[order.index(l) for l in range(4)]
    a,t=state.A[c].clone(),state.B[b].clone()
    for kind,colour,x in (('A',c,a),('B',b,t)):
        for leg,w in enumerate(state.lambdas(kind,colour)):
            if leg!=k:
                shape=[1]*4;shape[leg+1]=D;x*=w.reshape(shape)
    ga,gb=cross_factors(float(dt))
    x=apply_local_factors(a.permute(order),ga,[i],6)
    y=apply_local_factors(t.permute(order),gb,[j],3)
    w=state.weights[c][k].sqrt().reshape(1,1,1,D,1)
    x=(x*w).reshape(64*D*D,-1);y=(y*w).reshape(8*D*D,-1)
    qx,rx=torch.linalg.qr(x,mode='reduced');qy,ry=torch.linalg.qr(y,mode='reduced')
    u,s,vh=torch.linalg.svd(rx@ry.T,full_matrices=False)
    discard=float(s[D:].square().sum()/s.square().sum())
    a=(qx@u[:,:D]).reshape(64,D,D,D).permute(inverse)
    t=(qy@vh[:D].T).reshape(8,D,D,D).permute(inverse)
    new_w=s[:D]/s[:D].norm();change=float((new_w-state.weights[c][k]).abs().max())
    for kind,colour,x in (('A',c,a),('B',b,t)):
        for leg,w in enumerate(state.lambdas(kind,colour)):
            if leg!=k:
                shape=[1]*4;shape[leg+1]=D;x/=w.clamp_min(1e-12).reshape(shape)
    state.A[c],state.B[b]=a/a.norm(),t/t.norm()
    state.weights[c][k]=new_w
    return change,discard


def orbit_representatives():
    # A/B onsite field groups commute internally and are exact block gates.
    return [('field','A'),('field','B')]+[('internal',i) for i in range(6)]+[
        ('cross',k,i,j) for k in range(3) for i,j in edge_bonds(k)]


@lru_cache(None)
def field_diagonal(kind,field,dt):
    n=6 if kind=='A' else 3
    z=torch.tensor([n/2-index.bit_count() for index in range(2**n)],dtype=torch.float64)
    return torch.exp(float(dt)*float(field)*z)


@torch.no_grad()
def update_orbit(state,term,dt):
    """Unrestricted: three explicit disjoint gates. C3: one plus its copies."""
    result=[]
    for r in ((0,) if state.c3 else range(3)):
        if term[0]=='field':
            kind=term[1];group=state.A if kind=='A' else state.B
            c=r if kind=='A' else (-r)%3
            group[c]=group[c]*field_diagonal(kind,state.field,float(dt)).to(group[c]).reshape(-1,1,1,1)
            group[c]=group[c]/group[c].norm()
            if state.c3:copy_rotation_orbit(state,kind,c)
        elif term[0]=='internal':
            i=(term[1]-2*r)%6;j=(i+1)%6
            state.A[r]=apply_inside(state.A[r],physical_gate(float(dt)),[i,j],6)
            if state.c3:copy_rotation_orbit(state,'A',r)
        elif term[0]=='cross':
            _,k,i,j=term;k=(k-r)%3;i=(i-2*r)%6;j=(j-r)%3
            result.append(update_cross(state,r,k,i,j,dt))
            if state.c3:
                copy_rotation_orbit(state,'A',0)
                copy_rotation_orbit(state,'B',neighbour(0,k))
                for a in (1,2):state.weights[a][(k-a)%3]=state.weights[0][k].clone()
        else:
            raise ValueError('Unknown physical term')
    return max((x[0] for x in result),default=0.),max((x[1] for x in result),default=0.)


@torch.no_grad()
def sweep(state,dt,second_order=True):
    terms=orbit_representatives()
    sequence=terms+terms[::-1] if second_order else terms
    values=[update_orbit(state,t,dt/2 if second_order else dt) for t in sequence]
    state.steps+=1
    return max(v[0] for v in values),max(v[1] for v in values)
