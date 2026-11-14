"""Complex honeycomb CTM measurement, preserving Im<Sp_i Sm_j> for Dz.

The old real-observable function is untouched. One face at a time bounds the
number of large corner/response tensors. Intermediate non-Hermitian spin
operators return COMPLEX values; only assembled Hermitian observables are real.
"""
import torch
import math
from ...simple_update import SZ,SP,SM,embed
from ...geometry import face_nodes,neighbour,edge_bonds
from ...honeycomb_ctm import layers,double_layer,face_pairs,expanded_corner,FaceEnvironment,normalized,contract
from .model import arrow_sign,complex_state
from .linalg import full_svd


class DMIEnvironment(FaceEnvironment):
    """Same native CTM equations; exact CUDA gesvd instead of default gesvdj."""
    def grow(self,local,c,chi):
        aa,bb,ae=face_pairs(local,c)
        cp=[normalized(expanded_corner(self.C[i],self.Ta[i],self.Tb[(i-1)%3],aa[i],bb[i])) for i in range(3)]
        us=[]
        for i in range(3):
            u,_,_=full_svd(cp[(i+1)%3]@cp[(i+2)%3]@cp[i]);us.append(u[:,:min(chi,u.shape[1])])
        cc=[];ta=[];tb=[]
        for i in range(3):
            cc.append(normalized(us[(i-1)%3].mH@cp[i]@us[i]))
            ua=us[i].reshape(self.Tb[i].shape[0],ae[i].shape[0],-1)
            ub=us[i].reshape(self.Ta[i].shape[2],bb[i].shape[0],-1)
            ta.append(normalized(contract('lzL,zyx,lxr->Lyr',ua.conj(),ae[i],self.Tb[i])))
            tb.append(normalized(contract('lxr,zxk,rzK->lkK',self.Ta[i],bb[i],ub)))
        return DMIEnvironment(cc,ta,tb)

    def spectrum(self):
        _,s,_=full_svd(self.C[0]@self.C[1]@self.C[2])
        return s/s.norm().clamp_min(1e-300)


def environment(state,**settings):
    """Use the existing native CTM kernel with matching complex/device boundaries."""
    from ...honeycomb_ctm import HoneycombEnvironment
    complex_state(state)
    ref=state.A[0]
    faces=[DMIEnvironment.initial(state.D,dtype=ref.dtype) for _ in range(3)]
    for face in faces:
        face.C=[t.to(ref.device) for t in face.C]
        face.Ta=[t.to(ref.device) for t in face.Ta]
        face.Tb=[t.to(ref.device) for t in face.Tb]
    return HoneycombEnvironment(faces).run(layers(state),**settings)


class ComplexFaceMeasurement:
    def __init__(self,env,local,c):
        self.env,self.local,self.c=env,local,c
        self.cp=env.corners(local,c)
        self.norm=torch.trace(self.cp[0]@self.cp[1]@self.cp[2])
        if not torch.isfinite(self.norm) or abs(complex(self.norm))<1e-250:
            raise ArithmeticError('Invalid complex CTM overlap')
        self.responses=None

    def expectation(self,replacements):
        if len(replacements)==1:
            if self.responses is None:
                with torch.enable_grad():
                    variables={key:t.detach().requires_grad_() for key,t in self.local.items()}
                    cc=self.env.corners(variables,self.c)
                    norm=torch.trace(cc[0]@cc[1]@cc[2])
                    # For f(z)=a*z, grad Re(f) is conj(a). Undo that
                    # conjugation to recover the complex linear response.
                    grad=torch.autograd.grad(norm.real,list(variables.values()))
                    self.responses={key:g.conj().detach()/norm.detach() for key,g in zip(variables,grad)}
            key,value=next(iter(replacements.items()))
            return complex((self.responses[key]*value).sum())
        local=self.local|replacements;aa,bb,_=face_pairs(local,self.c);nodes=face_nodes(self.c)
        cc=[]
        for i in range(3):
            if nodes[2*i] in replacements or nodes[2*i+1] in replacements:
                cc.append(expanded_corner(self.env.C[i],self.env.Ta[i],self.env.Tb[(i-1)%3],aa[i],bb[i]))
            else:cc.append(self.cp[i])
        return complex(torch.trace(cc[0]@cc[1]@cc[2])/self.norm)


def _real(value):
    if abs(value.imag)>1e-8*max(1.,abs(value.real)):
        raise ArithmeticError(f'Complex result for a Hermitian observable: {value}')
    return value.real


def central_faces(c,k):
    b=neighbour(c,k);found=[]
    for f in range(3):
        nodes=face_nodes(f)
        for edge in range(6):
            if {nodes[edge],nodes[(edge+1)%6]}=={('A',c),('B',b)} and (2,0,1,2,0,1)[edge]==k:
                found.append(f)
    assert len(found)==2
    return found


@torch.no_grad()
def measure(state,env,*,J=1.,Dz=0.,strict=True):
    # strict=False is an explicitly labelled diagnostic path; it does not
    # certify the original 1e-8 test. Keep every maximum imaginary residual.
    max_imag=0.
    def checked(value):
        nonlocal max_imag
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            raise ArithmeticError('Non-finite observable')
        max_imag=max(max_imag,abs(value.imag))
        return _real(value) if strict else value.real
    local=layers(state);site_values={};bond_values={}
    for face in range(3):
        measurement=ComplexFaceMeasurement(env.faces[face],local,face)
        cache={}
        def impurity(kind,c,i,axis):
            key=(kind,c,i,axis)
            if key not in cache:
                op=embed((SZ,SP,SM)[axis],i,6 if kind=='A' else 3)
                cache[key]=double_layer(state.dressed(kind,c),op)
            return cache[key]
        for kind,n in (('A',6),('B',3)):
            for c in range(3):
                for i in range(n):
                    z,p,m=(measurement.expectation({(kind,c):impurity(kind,c,i,axis)}) for axis in range(3))
                    site_values.setdefault((kind,c,i),[]).append((checked((p+m)/2),checked((p-m)/2j),checked(z)))
                cache.clear()
        for c in range(3):
            for i in range(6):
                j=(i+1)%6;sign=arrow_sign('internal',-1,i,j)
                z,p,m=(embed(op,i,6).to(torch.complex128) for op in (SZ,SP,SM))
                zz=z@embed(SZ,j,6).to(z);pm=p@embed(SM,j,6).to(p);mp=m@embed(SP,j,6).to(m)
                ss=measurement.expectation({('A',c):double_layer(state.dressed('A',c),zz+.5*(pm+mp))})
                ch=measurement.expectation({('A',c):double_layer(state.dressed('A',c),sign*.5j*(pm-mp))})
                bond_values.setdefault(('internal',c,-1,i,c,j),[]).append((checked(ss),checked(ch)))
            for k in range(3):
                if face not in central_faces(c,k):continue
                b=neighbour(c,k)
                for i,j in edge_bonds(k):
                    def cross(a1,a2):
                        return measurement.expectation({('A',c):impurity('A',c,i,a1),('B',b):impurity('B',b,j,a2)})
                    zz,pm,mp=cross(0,0),cross(1,2),cross(2,1)
                    ss=zz+.5*(pm+mp);ch=arrow_sign('cross',k,i,j)*.5j*(pm-mp)
                    bond_values.setdefault(('cross',c,k,i,b,j),[]).append((checked(ss),checked(ch)))
                cache.clear()
        del measurement
    sites=[];bonds=[];consistency=0.
    for (kind,c,i),values in site_values.items():
        sx,sy,sz=(sum(v[k] for v in values)/3 for k in range(3))
        sites.append(dict(kind=kind,colour=c,site=i,sx=sx,sy=sy,sz=sz,
                          transverse_moment=(sx*sx+sy*sy)**.5))
        consistency=max(consistency,max(v[2] for v in values)-min(v[2] for v in values))
    for key,values in bond_values.items():
        kind,a,k,i,b,j=key;n=len(values)
        assert n==(3 if kind=='internal' else 2)
        ss,ch=(sum(v[k] for v in values)/n for k in range(2))
        bonds.append(dict(kind=kind,a=a,k=k,i=i,b=b,j=j,s_dot_s=ss,
            vector_chirality_z_ccw=ch,energy_exchange=J*ss,energy_dmi=Dz*ch,energy=J*ss+Dz*ch))
        consistency=max(consistency,max(v[1] for v in values)-min(v[1] for v in values))
    assert len(sites)==27 and len(bonds)==54
    mz=sum(s['sz'] for s in sites)/27;exchange=sum(b['energy_exchange'] for b in bonds)/27
    dm=sum(b['energy_dmi'] for b in bonds)/27
    return dict(D=state.D,J=J,Dz=Dz,field=state.field,energy_exchange=exchange,energy_dmi=dm,
        energy_field=-state.field*mz,energy_total=exchange+dm-state.field*mz,
        mean_sz=mz,magnetization_normalized=2*mz,sites=sites,bonds=bonds,
        max_transverse_moment=max(s['transverse_moment'] for s in sites),
        face_consistency_max=consistency,ctm_converged=env.converged,
        ctm_iterations=env.iterations,ctm_residual=env.residual if math.isfinite(env.residual) else None,
        max_hermitian_imaginary_part=max_imag,hermiticity_ok_1e8=max_imag<1e-8,strict_measurement=strict,
        convention='CCW every triangle; H=JSS+Dz*kappa-hSz; complex dense tensors')
