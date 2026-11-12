"""Conservative, lambda-compatible gauge alignment before optional C3 reduction.

Gauge rotations are confined to equal-lambda subspaces; signs are fixed
covariantly using a largest-coefficient pivot. These transformations leave
the represented network unchanged. Only the subsequent orbit averaging is
an approximation, with tensor, lambda and local-density-proxy acceptance
tests. A failed test returns the ORIGINAL six-tensor state, never forces C3.
This explicit reduction protocol is our choice: the paper does not give it.
"""
import copy
import torch
from ..simple_update import rotate_tensor
from ..geometry import neighbour


def rotate(t,kind,n):
    for _ in range(n%3):t=rotate_tensor(t,kind)
    return t


def mode_product(t,g,k):
    axis=k+1;order=[i for i in range(4) if i!=axis]+[axis]
    x=t.permute(order);shape=x.shape
    x=(x.reshape(-1,shape[-1])@g).reshape(shape)
    return x.permute([order.index(i) for i in range(4)]).contiguous()


def groups(w,tolerance):
    result=[];start=0
    for end in range(1,len(w)+1):
        if end==len(w) or abs(float(w[end]-w[start]))>tolerance*max(float(w.max()),1e-30):
            result.append(list(range(start,end)));start=end
    return result


@torch.no_grad()
def canonical_dressed(state,degeneracy_tolerance=1e-10):
    a=[state.dressed('A',c).clone() for c in range(3)]
    b=[state.dressed('B',c).clone() for c in range(3)]
    for c in range(3):
        for k in range(3):
            # HOSVD fixes continuous gauge freedom ONLY within equal weights.
            x=a[c].movedim(k+1,-1).reshape(-1,state.D)
            gram=x.mT@x;g=torch.eye(state.D,dtype=x.dtype)
            for inds in groups(state.weights[c][k],degeneracy_tolerance):
                if len(inds)>1:
                    idx=torch.tensor(inds);_,u=torch.linalg.eigh(gram[idx[:,None],idx])
                    g[idx[:,None],idx]=u.flip(1)
            a[c]=mode_product(a[c],g,k)
            target=neighbour(c,k);b[target]=mode_product(b[target],g,k)
        # Pivot selection depends on absolute values, hence is covariant under
        # physical/virtual permutations and arbitrary diagonal sign gauges.
        pivot=list(torch.unravel_index(a[c].abs().argmax(),a[c].shape))
        signs=[]
        for k in range(3):
            sl=pivot.copy();sl[k+1]=slice(None)
            v=a[c][tuple(sl)];signs.append(torch.where(v<0,-torch.ones_like(v),torch.ones_like(v)))
        for k,v in enumerate(signs):
            g=torch.diag(v);a[c]=mode_product(a[c],g,k)
            target=neighbour(c,k);b[target]=mode_product(b[target],g,k)
    for group in (a,b):
        for c,t in enumerate(group):
            pivot=t.flatten()[t.abs().argmax()]
            group[c]=t if pivot>=0 else -t # irrelevant per-block scalar phase
    return a,b


def rho(t):
    x=t.flatten(1);value=x@x.mT
    return value/torch.trace(value)


@torch.no_grad()
def try_reduce(state,tensor_tolerance=1e-3,weight_tolerance=1e-3,
               rho_tolerance=1e-4,degeneracy_tolerance=1e-10):
    if state.c3:
        return state,dict(accepted=True,reason='already_C3',gauge_aligned_tensor_residual=0.)
    a,b=canonical_dressed(state,degeneracy_tolerance)
    means={kind:sum(rotate(group[c],kind,(-c if kind=='A' else c)) for c in range(3))/3
           for kind,group in (('A',a),('B',b))}
    for key,t in means.items():
        if float(t.norm())<1e-12:
            return state,dict(accepted=False,reason='orbit_average_cancels')
        means[key]=t/t.norm()
    residual=max(float((group[c]-rotate(means[kind],kind,c if kind=='A' else -c)).norm())
        for kind,group in (('A',a),('B',b)) for c in range(3))
    orbit_weights=[];weight_residual=0.
    for k in range(3):
        ws=[state.weights[c][(k-c)%3] for c in range(3)]
        avg=sum(ws)/3;avg=avg/avg.norm()
        weight_residual=max(weight_residual,max(float((w-avg).abs().max()) for w in ws))
        orbit_weights.append(avg.clamp_min(1e-14)/avg.clamp_min(1e-14).norm())
    candidate=copy.deepcopy(state);candidate.c3=True
    candidate.weights=[[orbit_weights[(c+k)%3].clone() for k in range(3)] for c in range(3)]
    for kind,group in (('A',candidate.A),('B',candidate.B)):
        for c in range(3):
            t=rotate(means[kind],kind,c if kind=='A' else -c).clone()
            for k,w in enumerate(candidate.lambdas(kind,c)):
                shape=[1]*4;shape[k+1]=len(w);t/=w.sqrt().reshape(shape)
            group[c]=t/t.norm()
    # This is a product-environment proxy, explicitly NOT an infinite-lattice
    # observable or fidelity certificate. It supplements the tensor bounds.
    rho_drift=max(float((rho(state.dressed(kind,c,power=1))-
                         rho(candidate.dressed(kind,c,power=1))).norm())
                  for kind in ('A','B') for c in range(3))
    accepted=(residual<=tensor_tolerance and weight_residual<=weight_tolerance and rho_drift<=rho_tolerance)
    info=dict(accepted=accepted,reason='within_all_bounds' if accepted else 'not_close_enough_keep_six',
        gauge_aligned_tensor_residual=residual,lambda_orbit_residual=weight_residual,
        rho_proxy_drift=rho_drift,tensor_tolerance=tensor_tolerance,
        weight_tolerance=weight_tolerance,rho_tolerance=rho_tolerance)
    return (candidate if accepted else state),info
