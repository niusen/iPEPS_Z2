from itertools import groupby
from functools import reduce

import numpy,torch,yastn


# function my_pinv(T::DiagonalTensorMap)
#     epsilon0 = 1e-12
#     epsilon=epsilon0*maximum(abs.(diag(convert(Array,T))))^2
#     T_new=deepcopy(T);
    
#     mm=T_new.data;
#     mm = mm./(mm.^2 .+epsilon)
#     T_new.data.=mm;

#     return T_new
# end



# function my_tsvd(T::AbstractTensorMap; kwargs...)
#     (U,S,V) = tsvd(T;kwargs...);
#     return (U,S,V)
# end
# function ChainRulesCore.rrule(::typeof(my_tsvd), t::AbstractTensorMap;kwargs...)
#     global multiplet_tol, backward_settings
#     #grad_inverse_tol=1e-8
#     #grad_regulation_epsilon=1e-12;
#     (U,S,V) = my_tsvd(t;kwargs...);#println(S.data.values)
#     epsilon1=maximum(diag(convert(Array,S)))*backward_settings.grad_inverse_tol;
#     epsilon2=maximum(diag(convert(Array,S)))*backward_settings.grad_regulation_epsilon;
#     Fp = similar(S);
#     for (k,dst) in blocks(Fp)
#         src = blocks(S)[k]
#         @inbounds for i in 1:size(dst,1),j in 1:size(dst,2)
#             ff=0;
#             if abs(src[j,j]-src[i,i])>epsilon1
#                 if abs(abs(src[j,j])/abs(src[i,i])-1)>multiplet_tol #relative difference is big
#                     ff=ff + (src[j,j]-src[i,i])/((src[j,j]-src[i,i])^2+epsilon2)
#                 end
#             end
#             if src[j,j] + src[i,i]>epsilon1
#                 ff=ff + (src[j,j] + src[i,i])/((src[j,j] + src[i,i])^2+epsilon2)
#             end
#             dst[i,j] = (i == j) ? zero(eltype(S)) : ff
#         end
#     end

#     Fm = similar(S);
#     for (k,dst) in blocks(Fm)
#         src = blocks(S)[k]
#         @inbounds for i in 1:size(dst,1),j in 1:size(dst,2)
#             ff=0;
#             if abs(src[j,j]-src[i,i])>epsilon1
#                 if abs(abs(src[j,j])/abs(src[i,i])-1)>multiplet_tol #relative difference is big
#                     ff=ff + (src[j,j]-src[i,i])/((src[j,j]-src[i,i])^2+epsilon2)
#                 end
#             end
#             if src[j,j] + src[i,i]>epsilon1
#                 ff=ff - (src[j,j] + src[i,i])/((src[j,j] + src[i,i])^2+epsilon2)
#             end
#             dst[i,j] = (i == j) ? zero(eltype(S)) : ff
#         end
#     end
#     #jldsave("svd.jld2"; U,S,V)

#     function pullback(v)
#         dU,dS,dV = v
#         #jldsave("svd_backward.jld2"; dU,dS,dV)
#         #println("Norm of dU, dS, dV: "*string([norm(dU),norm(dS),norm(dV)]))

#         dA = zero(t);
#         #A_s bar term
#         if dS != ChainRulesCore.ZeroTangent()
#             dA += U*_elementwise_mult(dS,one(dS))*V
#         end
#         #A_uo bar term
#         if dU != ChainRulesCore.ZeroTangent()
#             Jp = _elementwise_mult((U'*dU),Fp) - _elementwise_mult(dU'*U,Fp)
            
#             dA += U*(Jp)*V/2
#         end
#         #A_vo bar term
#         if dV != ChainRulesCore.ZeroTangent()
#             VpdV = V*dV';
#             Km = _elementwise_mult(VpdV,Fm) - _elementwise_mult(dV*(V'),Fm)
#             dA += U*(Km)*V/2
#         end

#         # ####!!!  In my test without such term is more accurate.
#         # #A_d bar term, only relevant if matrix is complex
#         # if dV != ChainRulesCore.ZeroTangent() && T <: Complex
#         #     L = _elementwise_mult(V*dV',one(Fm))
#         #     dA += 1/2*U*my_pinv(S)*(L' - L)*V
#         # end

#         if codomain(t)!=domain(t)
#             pru = U*U';
#             prv = V'*V;
#             dA += (one(pru)-pru)*dU*my_pinv(S)*V
#             dA += U*my_pinv(S)*dV*(one(prv)-prv)
#         end

#         if backward_settings.show_ite_grad_norm
#             println("Norm of dA: "*string(norm(dA)))
#         end
#         global grad_norm
#         grad_norm=deepcopy(norm(dA));
#         return NoTangent(), dA, [NoTangent() for kwa in kwargs]...
#     end
#     return (U,S,V), pullback
# end


def safe_inverse(x, eps_rel=1.0e-12, eps_abs=1.0e-12):
    return x / (x ** 2 + eps_abs)


# def safe_inverse_2(x, epsilon):
#     x[abs(x) < epsilon] = float('inf')
#     return x.pow(-1)


class SVDGESDD(torch.autograd.Function):
    @staticmethod
    def forward(A, ad_decomp_reg, fullrank_uv, diagnostics):
        U, S, Vh = torch.linalg.svd(A, full_matrices=fullrank_uv)
        # A = U @ diag(S) @ Vh
        return U, S, Vh

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        _, ad_decomp_reg, _, diagnostics= inputs
        U, S, Vh= output
        ctx.save_for_backward(U, S, Vh, ad_decomp_reg)
        ctx.diagnostics= diagnostics


    @staticmethod
    def backward(self, gu, gsigma, gvh):
        r"""
        :param gu: gradient on U
        :type gu: torch.Tensor
        :param gsigma: gradient on S
        :type gsigma: torch.Tensor
        :param gv: gradient on V
        :type gv: torch.Tensor
        :return: gradient
        :rtype: torch.Tensor

        Computes backward gradient for SVD, adopted from
        https://github.com/pytorch/pytorch/blob/v1.10.2/torch/csrc/autograd/FunctionsManual.cpp

        For complex-valued input there is an additional term, see

            * https://giggleliu.github.io/2019/04/02/einsumbp.html
            * https://arxiv.org/abs/1909.02659

        The backward is regularized following

            * https://github.com/wangleiphy/tensorgrad/blob/master/tensornets/adlib/svd.py
            * https://arxiv.org/abs/1903.09650

        using

        .. math::
            S_i/(S^2_i-S^2_j) = (F_{ij}+G_{ij})/2\ \ \textrm{and}\ \ S_j/(S^2_i-S^2_j) = (F_{ij}-G_{ij})/2

        where

        .. math::
            F_{ij}=1/(S_i-S_j),\ G_{ij}=1/(S_i+S_j)
        """
        #
        # TORCH_CHECK(compute_uv,
        #    "svd_backward: Setting compute_uv to false in torch.svd doesn't compute singular matrices, ",
        #    "and hence we cannot compute backward. Please use torch.svd(compute_uv=True)");


        diagnostics= self.diagnostics
        u, sigma, vh, eps= self.saved_tensors
        m= u.size(0) # first dim of original tensor A = u sigma v^\dag
        n= vh.size(1) # second dim of A
        k= sigma.size(0)
        sigma_scale= sigma[0]

        # ? some
        if (u.size(-2)!=u.size(-1)) or (vh.size(-2)!=vh.size(-1)):
            # We ignore the free subspace here because possible base vectors cancel
            # each other, e.g., both -v and +v are valid base for a dimension.
            # Don't assume behavior of any particular implementation of svd.
            u = u.narrow(-1, 0, k)
            vh = vh.narrow(-2, 0, k)
            if not (gu is None): gu = gu.narrow(-1, 0, k)
            if not (gvh is None): gvh = gvh.narrow(-2, 0, k)


        if not (gsigma is None):
            # computes u @ diag(gsigma) @ vh
            sigma_term = u * gsigma.unsqueeze(-2) @ vh
        else:
            sigma_term = torch.zeros(m,n,dtype=u.dtype,device=u.device)
        # in case that there are no gu and gvh, we can avoid the series of kernel
        # calls below
        if (gu is None) and (gvh is None):
            if not (diagnostics is None):
                print(f"{diagnostics} {sigma_term.abs().max()} {sigma.max()}")
            return sigma_term, None, None, None

        # sigma_inv= safe_inverse_2(sigma.clone(), sigma_scale*eps)
        # sigma_inv= safe_inverse(sigma.clone(), eps_abs=sigma_scale*eps)
        sigma_inv= safe_inverse(sigma.clone(), eps)

        F = sigma.unsqueeze(-2) - sigma.unsqueeze(-1)
        F = safe_inverse(F, sigma_scale*eps)
        F.diagonal(0,-2,-1).fill_(0)

        G = sigma.unsqueeze(-2) + sigma.unsqueeze(-1)
        G = safe_inverse(G, sigma_scale*eps)
        G.diagonal(0,-2,-1).fill_(0)

        uh= u.conj().transpose(-2,-1)
        if not (gu is None):
            guh = gu.conj().transpose(-2, -1);
            u_term = u @ ( (F+G).mul( uh @ gu - guh @ u) ) * 0.5
            if m > k:
                # projection operator onto subspace orthogonal to span(U) defined as I - UU^H
                proj_on_ortho_u = -u @ uh
                proj_on_ortho_u.diagonal(0, -2, -1).add_(1);
                u_term = u_term + proj_on_ortho_u @ (gu * sigma_inv.unsqueeze(-2))
            u_term = u_term @ vh
        else:
            u_term = torch.zeros(m,n,dtype=u.dtype,device=u.device)

        v= vh.conj().transpose(-2,-1)
        if not (gvh is None):
            gv = gvh.conj().transpose(-2, -1);
            v_term = ( (F-G).mul(vh @ gv - gvh @ v) ) @ vh * 0.5
            if n > k:
                # projection operator onto subspace orthogonal to span(V) defined as I - VV^H
                proj_on_v_ortho =  -v @ vh
                proj_on_v_ortho.diagonal(0, -2, -1).add_(1);
                v_term = v_term + sigma_inv.unsqueeze(-1) * (gvh @ proj_on_v_ortho)
            v_term = u @ v_term
        else:
            v_term = torch.zeros(m,n,dtype=u.dtype,device=u.device)

        # TODO enable check
        # if (u.is_complex() or vh.is_complex()) and not (gvh is None) and not (gu is None):
        #     imdiag_UhgU= UhgU.diagonal(0, -2, -1).imag
        #     imdiag_VhgV= VhgV.diagonal(0, -2, -1).imag
        #     if not torch.allclose( imdiag_UhgU, -imdiag_VhgV, 1e-2, 1e-2 ):
        #         warnings.warn("svd_backward: The singular vectors in the complex case are "\
        #         +"specified up to multiplication by e^{i phi}. The specified loss function depends on "\
        #         +"this phase term, making it ill-defined.",RuntimeWarning)

        # // for complex-valued input there is an additional term
        # // https://giggleliu.github.io/2019/04/02/einsumbp.html
        # // https://arxiv.org/abs/1909.02659
        dA= u_term + sigma_term + v_term
        if u.is_complex() or v.is_complex():
            L= (uh @ gu).diagonal(0,-2,-1)
            L.real.zero_()
            L.imag.mul_(sigma_inv)
            imag_term= (u * L.unsqueeze(-2)) @ vh
            dA= dA + imag_term

        if diagnostics is not None:
            print(f"{diagnostics} {dA.abs().max()} {sigma.max()}")

        return dA, None, None, None

class kernel_svd(torch.autograd.Function):
    @staticmethod
    def forward(data, meta, sizes, fullrank_uv=False, ad_decomp_reg=1.0e-12, diagnostics=None):
        real_dtype = data.real.dtype if data.is_complex() else data.dtype
        Udata = torch.empty((sizes[0],), dtype=data.dtype, device=data.device)
        Sdata = torch.empty((sizes[1],), dtype=real_dtype, device=data.device)
        Vhdata = torch.empty((sizes[2],), dtype=data.dtype, device=data.device)
        reg = torch.as_tensor(ad_decomp_reg, dtype=real_dtype, device=data.device)
        for (sl, D, slU, DU, slS, slV, DV) in meta:
            U, S, Vh = SVDGESDD.forward(data[slice(*sl)].view(D), reg, fullrank_uv, diagnostics)
            Udata[slice(*slU)].reshape(DU)[:] = U
            Sdata[slice(*slS)] = S
            Vhdata[slice(*slV)].reshape(DV)[:] = Vh
        return Udata, Sdata, Vhdata

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        data, meta, sizes, _, ad_decomp_reg,diagnostics= inputs
        reg= torch.as_tensor(ad_decomp_reg, dtype=data.real.dtype, device=data.device)
        Udata, Sdata, Vhdata= output
        ctx.save_for_backward(Udata, Sdata, Vhdata, reg)
        ctx.meta_svd= meta
        ctx.data_size= data.numel()
        ctx.diagnostics= diagnostics

    @staticmethod
    def backward(ctx, Udata_b, Sdata_b, Vhdata_b):
        Udata, Sdata, Vhdata, reg= ctx.saved_tensors
        meta= ctx.meta_svd
        diagnostics= ctx.diagnostics
        data_size= ctx.data_size
        data_b= torch.zeros(data_size, dtype=Udata.dtype, device=Udata.device)
        for (sl, D, slU, DU, slS, slV, DV) in meta:
            loc_ctx= SimpleNamespace(diagnostics=diagnostics,
                saved_tensors=(Udata[slice(*slU)].view(DU),Sdata[slice(*slS)],Vhdata[slice(*slV)].view(DV),reg))
            data_b[slice(*sl)].view(D)[:],_,_,_ = SVDGESDD.backward(loc_ctx,\
                Udata_b[slice(*slU)].view(DU),Sdata_b[slice(*slS)],Vhdata_b[slice(*slV)].view(DV))
        return data_b, None, None, None, None, None
    
def svd(data, meta, sizes, fullrank_uv=False, ad_decomp_reg=1.0e-12, diagnostics=None, **kwargs):
    return kernel_svd.apply(data, meta, sizes, fullrank_uv, ad_decomp_reg, diagnostics)


def svd_with_truncation(a, axes=(0, 1), sU=1, nU=True,
        Uaxis=-1, Vaxis=0, policy='fullrank', fix_signs=False, svd_on_cpu=False,
        tol=0, tol_block=0, D_block=float('inf'), D_total=float('inf'),
        truncate_multiplets=False, mask_f=None, **kwargs) -> tuple[yastn.Tensor, yastn.Tensor, yastn.Tensor]:
    r"""
    Split tensor using exact singular value decomposition (SVD) into :math:`a = U S V`,
    where the columns of `U` and the rows of `V` form orthonormal bases
    and `S` is positive and diagonal matrix.

    The function allows for optional truncation.
    Truncation can be based on relative tolerance, bond dimension of each block,
    and total bond dimension across all blocks (whichever gives smaller total dimension).

    Parameters
    ----------
    axes: tuple[int, int] | tuple[Sequence[int], Sequence[int]]
        Specify two groups of legs between which to perform SVD, as well as
        their final order.

    sU: int
        signature of the new leg in U; equal 1 or -1. The default is 1.
        V is going to have opposite signature on connecting leg.

    nU: bool
        Whether or not to attach the charge of  ``a`` to `U`.
        If ``False``, it is attached to `V`. The default is ``True``.

    Uaxis, Vaxis: int
        specify which leg of `U` and `V` tensors are connecting with `S`. By default,
        it is the last leg of `U` and the first of `V`.

    policy: str
        ``"fullrank"`` or ``"lowrank"`` are allowed. For ``"fullrank"`` use standard full (but reduced) SVD,
        and for ``"lowrank"`` use randomized/truncated SVD and requires providing ``D_block`` in ``kwargs``.

    tol: float
        relative tolerance of singular values below which to truncate across all blocks.

    tol_block: float
        relative tolerance of singular values below which to truncate within individual blocks.

    D_block: int
        largest number of singular values to keep in a single block.

    D_total: int
        largest total number of singular values to keep.

    truncate_multiplets: bool
        If ``True``, enlarge the truncation range specified by other arguments by shifting
        the cut to the largest gap between to-be-truncated singular values across all blocks.
        It provides a heuristic mechanism to avoid truncating part of a multiplet.
        The default is ``False``.

    mask_f: function[yastn.Tensor] -> yastn.Tensor
        custom truncation-mask function.
        If provided, it overrides all other truncation-related arguments.

    Returns
    -------
    `U`, `S`, `V`
    """
    diagnostics = kwargs.get('diagonostics', None)
    U, S, V = svd(a, axes=axes, sU=sU, nU=nU, policy=policy, D_block=D_block,
                  diagnostics=diagnostics, fix_signs=fix_signs, svd_on_cpu=svd_on_cpu)

    Smask = truncation_mask(S, tol=tol, tol_block=tol_block,
                            D_block=D_block, D_total=D_total,
                            truncate_multiplets=truncate_multiplets,
                            mask_f=mask_f)

    U, S, V = Smask.apply_mask(U, S, V, axes=(-1, 0, 0))

    U = U.moveaxis(source=-1, destination=Uaxis)
    V = V.moveaxis(source=0, destination=Vaxis)
    return U, S, V
