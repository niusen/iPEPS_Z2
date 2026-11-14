"""H=J sum S_i.S_j + Dz sum_CCW (S_i x S_j)_z - h sum Sz.

Every elementary triangle is counterclockwise in geometry.py Cartesian x/y.
Dz is signed; reversing an ordered bond changes its DMI sign. J=0 is allowed.
Only out-of-plane DMI is included, so [H,sum Sz]=0. Dz=0 is the old model.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import torch
from ...geometry import A1,A2,points,SHIFTS,edge_bonds
from ...simple_update import SZ,SP,SM,BlockState


@dataclass(frozen=True)
class Couplings:
    J: float=1.
    Dz: float=0.
    h: float=.38

    def __post_init__(self):
        if not all(np.isfinite(x) for x in (self.J,self.Dz,self.h)):
            raise ValueError('Non-finite coupling')


def _on_kagome(p):
    basis=(-A1/2,A1*0,-A2/2);matrix=np.column_stack((A1,A2))
    return any(np.max(np.abs((r:=np.linalg.solve(matrix,p-b))-np.rint(r)))<1e-8 for b in basis)


@lru_cache(None)
def arrow_sign(kind,k,i,j):
    a=points('A')[i]
    b=points('A')[j] if kind=='internal' else points('B',*SHIFTS[k])[j]
    if abs(np.linalg.norm(a-b)-.5)>1e-9:raise ValueError('Not a nearest-neighbour bond')
    delta=b-a;normal=np.array([-delta[1],delta[0]])*np.sqrt(3)/2
    candidates=[p for p in ((a+b)/2+normal,(a+b)/2-normal) if _on_kagome(p)]
    if len(candidates)!=1:raise ValueError('Bond must belong to exactly one elementary triangle')
    v=candidates[0]-a;cross=delta[0]*v[1]-delta[1]*v[0]
    return 1 if cross>0 else -1


def bond_terms():
    return [('internal',-1,i,(i+1)%6) for i in range(6)]+[
        ('cross',k,i,j) for k in range(3) for i,j in edge_bonds(k)]


@lru_cache(None)
def two_spin_hamiltonian(J=1.,Dz=0.,sign=1):
    z,p,m=(a.to(torch.complex128) for a in (SZ,SP,SM))
    return J*torch.kron(z,z)+(J/2+.5j*Dz*sign)*torch.kron(p,m)+(J/2-.5j*Dz*sign)*torch.kron(m,p)


@lru_cache(None)
def physical_gate(dt,J=1.,Dz=0.,sign=1):
    return torch.matrix_exp(-float(dt)*two_spin_hamiltonian(float(J),float(Dz),int(sign)))


def complex_state(state):
    state.A=[a.to(torch.complex128) for a in state.A]
    state.B=[a.to(torch.complex128) for a in state.B]
    return state


def product_120(field=0.,chirality=1):
    """q=0 coherent reference; NOT an exact quantum ground-state energy.

    Positive chirality here means the lower-energy choice for Dz>0 and our
    CCW arrows. B1/B2/B3 angles are 0,120,240 degrees. No Sz or C3 projection.
    """
    state=complex_state(BlockState.random(1,field=field,c3=False))
    basis=(-A1/2,A1*0,-A2/2);matrix=np.column_stack((A1,A2))
    for kind,n in (('A',6),('B',3)):
        value=torch.ones(1,dtype=torch.complex128)
        for site in range(n):
            p=points(kind)[site]
            labels=[s for s,b in enumerate(basis) if np.max(np.abs((r:=np.linalg.solve(matrix,p-b))-np.rint(r)))<1e-8]
            if len(labels)!=1:raise ValueError('Physical sublattice not found')
            theta=chirality*2*np.pi*labels[0]/3
            spinor=torch.tensor([1,np.exp(1j*theta)],dtype=torch.complex128)/np.sqrt(2)
            value=torch.kron(value,spinor)
        group=state.A if kind=='A' else state.B
        for c in range(3):group[c]=value.reshape(2**n,1,1,1).clone()
    return state


def save_state(path,state,couplings,metadata=None):
    from pathlib import Path
    if state.field!=couplings.h:raise ValueError('State field and saved Hamiltonian h disagree')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(dict(format='kagome_hexagon_triangle_dmi_v1',A=state.A,B=state.B,weights=state.weights,
        field=state.field,seed=state.seed,steps=state.steps,c3=state.c3,
        couplings=dict(J=couplings.J,Dz=couplings.Dz,h=couplings.h),
        convention='CCW every elementary triangle; field=-h*Sz',metadata=metadata or {}),tmp)
    tmp.replace(path)


def load_state(path):
    value=torch.load(path,map_location='cpu',weights_only=False)
    if value['format']=='kagome_hexagon_triangle_v1':
        state=BlockState.load(path);return state,Couplings(h=state.field),{}
    if value['format']!='kagome_hexagon_triangle_dmi_v1':raise ValueError('Unknown checkpoint format')
    state=BlockState(**{k:value[k] for k in ('A','B','weights','field','seed','steps','c3')})
    return state,Couplings(**value['couplings']),value.get('metadata',{})
