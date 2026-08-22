import random
import yastn
import numpy,torch
from ansatz.triangle_iPESS import *
from ctmrg.Fermionic_CTMRG_unitcell_iPESS import *
from model.fermion_ob_iPESS import *
from config.settings import *
from config.config import *
import time


def _to_float(x):
    if hasattr(x, "item"):
        x = x.item()
    return float(numpy.real(x))


def _ctmrg_err_to_float(ite_err):
    if isinstance(ite_err, (list, tuple)):
        if len(ite_err)==0:
            return numpy.inf
        return max(_ctmrg_err_to_float(err) for err in ite_err)
    return _to_float(ite_err)


def _save_line_search_tensor_if_strict(state, E_trial, history_min_E, ite_err, D, chi, config_kwargs):
    E_trial_float=_to_float(E_trial)
    history_min_E_float=_to_float(history_min_E)
    ite_err_float=_ctmrg_err_to_float(ite_err)
    if (ite_err_float<1e-2) and (E_trial_float<history_min_E_float):
        filenm='Z2_D'+str(D)+'_chi'+str(chi)
        save_triangle_iPESS(state.B_set, state.T_set, filenm, config_kwargs)
        return True
    print('skip saving tensor: CTMRG err='+str(ite_err_float)+', E='+str(E_trial_float)+', history min E='+str(history_min_E_float))
    return False


def _clear_tensor_grad(tensor):
    if getattr(tensor, "_data", None) is not None and tensor._data.grad is not None:
        tensor._data.grad = None


def _clear_state_grads(state):
    for key in state.B_set:
        _clear_tensor_grad(state.B_set[key])
        _clear_tensor_grad(state.T_set[key])


def _assert_finite_grad(state_grad):
    grad_norm = state_grad.norm()
    if not numpy.isfinite(grad_norm):
        raise FloatingPointError("non-finite gradient norm")


def clip_state_grad(grad, max_norm):
    if max_norm is None or not numpy.isfinite(max_norm) or max_norm <= 0:
        return grad
    norm = grad.norm()
    if norm > max_norm:
        return scale_state(grad, max_norm / (norm + 1.0e-30))
    return grad


def random_tensor_sign(T):
    T=T.copy();
    data_1d,bb=yastn.Tensor.compress_to_1d(T);
    def generate_number(a):
        if a>0:
            b=random.random();
        elif a<0:
            b=-random.random();
        elif a==0:
            b=0;
        return b
    
    for dd in range(0,len(data_1d)):
        a=data_1d[dd].item();
        if data_1d.dtype==torch.float64:
            a_new=generate_number(a);
        elif data_1d.dtype==torch.complex128:
            a_new=generate_number(numpy.real(a))+1j*generate_number(numpy.imag(a));
        else:
            raise ValueError("unknown number type")
        
        data_1d[dd]=a_new;
    T=yastn.decompress_from_1d(data_1d,bb)

    return T


def get_random_grad(x0,delta):
    x=x0.copy();

    B_set_new=OrderedDict()
    T_set_new=OrderedDict()
    for key in x.B_set:
        B_set_new.update({key: random_tensor_sign(x.B_set[key])*delta})
        T_set_new.update({key: random_tensor_sign(x.T_set[key])*delta})

    return IPESS_TRIANGLE(B_set_new, T_set_new, x.global_args)

####################################

def cost_fun(parameters,state, ctm_args, energy_setting, global_args, config_kwargs, return_ctm_err=False):
    _clear_state_grads(state)
    state.require_grad(True)
    B_set=state.B_set
    T_set=state.T_set
    init=INITCTMARGS()
    CTM0=None;
    CTM_cell, double_B_set,double_T_set,ite_num,ite_err=Fermionic_CTMRG_cell_iPESS(B_set,T_set,init,CTM0, ctm_args, global_args);
    if (ctm_args.doublelayer_on_cpu)&(global_args.device !=double_B_set['1,1'].device) :
        double_B_set=Cell_to_device(double_B_set,global_args.device,global_args);
        double_T_set=Cell_to_device(double_T_set,global_args.device,global_args);
    if energy_setting.model=="spinful_triangle_lattice":
        E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell_iPESS(parameters, B_set,T_set, double_B_set, double_T_set, CTM_cell, energy_setting, config_kwargs, global_args);
    elif energy_setting.model in ("triangle_spinHall", "triangle_spinfulHofstadter"):
        E_total,  ex_up_set, ey_up_set, e_diagonala_up_set, ex_dn_set, ey_dn_set, e_diagonala_dn_set, e0_set, eU_set, sx_set, sy_set, sz_set=evaluate_ob_cell_iPESS(parameters, B_set,T_set, double_B_set, double_T_set, CTM_cell, energy_setting, config_kwargs, global_args);
    elif energy_setting.model == "triangle_spinlessHofstadter":
        E_total,  ex_set, ey_set, e_diagonala_set, e0_set =evaluate_ob_cell_iPESS(parameters, B_set,T_set, double_B_set, double_T_set, CTM_cell, energy_setting, config_kwargs, global_args);
    # print(E_total)
    # print(ex_set)
    # print(ey_set)
    # print(e_diagonala_set)
    # print(e0_set)
    # print(eU_set)
    print('E='+str(E_total.item()))
    if return_ctm_err:
        return E_total,CTM_cell,ite_err
    return E_total,CTM_cell
def get_grad(parameters,state, ctm_args, energy_setting, global_args, config_kwargs, return_ctm_err=False):
    if return_ctm_err:
        E, CTM_cell, ite_err=cost_fun(parameters,state, ctm_args, energy_setting, global_args, config_kwargs, return_ctm_err=True)
    else:
        E, CTM_cell=cost_fun(parameters,state, ctm_args, energy_setting, global_args, config_kwargs)
    start_time_grad = time.time()
    E.backward()
    end_grad = time.time()
    print('time on computing grad: '+time.strftime("%H hours, %M minuts, %S seconds", time.gmtime(end_grad - start_time_grad)))
    state.require_grad(False)

    B_set_grad=OrderedDict()
    T_set_grad=OrderedDict()
    for key in state.B_set:
        B_set_grad.update({key: state.B_set[key].grad()})
        T_set_grad.update({key: state.T_set[key].grad()})
    state_grad=IPESS_TRIANGLE(B_set_grad, T_set_grad, state.global_args);
    _assert_finite_grad(state_grad)
    CTM_cell=CTM_detach(CTM_cell, global_args)
    print("norm of grad:"+str(state_grad.norm()));
    if return_ctm_err:
        return state_grad,E,CTM_cell,ite_err
    return state_grad,E, CTM_cell


def state_axpy(state1, state2, coe1=1.0, coe2=1.0):
    B_set_new=OrderedDict()
    T_set_new=OrderedDict()
    for key in state1.B_set:
        B_set_new.update({key: state1.B_set[key]*coe1+state2.B_set[key]*coe2})
        T_set_new.update({key: state1.T_set[key]*coe1+state2.T_set[key]*coe2})
    return IPESS_TRIANGLE(B_set_new, T_set_new, state1.global_args)


def scale_state(state, coe):
    B_set_new=OrderedDict()
    T_set_new=OrderedDict()
    for key in state.B_set:
        B_set_new.update({key: state.B_set[key]*coe})
        T_set_new.update({key: state.T_set[key]*coe})
    return IPESS_TRIANGLE(B_set_new, T_set_new, state.global_args)


def add_scaled_state(state, direction, alpha):
    return state_axpy(state, direction, 1.0, alpha)


def state_inner(state1, state2):
    inner=0.0
    for key in state1.B_set:
        b1,_=yastn.Tensor.compress_to_1d(state1.B_set[key])
        b2,_=yastn.Tensor.compress_to_1d(state2.B_set[key])
        t1,_=yastn.Tensor.compress_to_1d(state1.T_set[key])
        t2,_=yastn.Tensor.compress_to_1d(state2.T_set[key])
        inner=inner+torch.sum(torch.conj(b1)*b2).real.item()
        inner=inner+torch.sum(torch.conj(t1)*t2).real.item()
    return inner


def make_descent_direction(grad, direction):
    directional_derivative=state_inner(grad,direction)
    if directional_derivative>=0:
        print("search direction is not descending; restart with steepest descent")
        direction=scale_state(grad,-1.0)
        directional_derivative=state_inner(grad,direction)
    return direction,directional_derivative

    
def subtract_state(state1,state2):
    return state_axpy(state1,state2,1.0,-1.0)



def fx(parameters,state, CTM0, ls_ctm_args, energy_setting, global_args, config_kwargs, return_ctm_err=False):
    state.require_grad(False)
    B_set=state.B_set
    T_set=state.T_set
    init=INITCTMARGS()
    init.reconstruct_CTM=False;
    
    CTM_cell, double_B_set,double_T_set,ite_num,ite_err=Fermionic_CTMRG_cell_iPESS(B_set,T_set,init,CTM0, ls_ctm_args, global_args);
    print('CTM ite_num='+str(ite_num)+', ite_err='+str(ite_err))

    if energy_setting.model=="spinful_triangle_lattice":
        E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell_iPESS(parameters, B_set,T_set, double_B_set, double_T_set, CTM_cell, energy_setting, config_kwargs, global_args);
        print('E= '+str(E_total.item()))
        print(ex_set.tolist())
        print(ey_set.tolist())
        print(e_diagonala_set.tolist())
        print(e0_set.tolist())
        print(eU_set.tolist())

        print('pairing:')
        pairing_x_set,pairing_y_set,pairing_diagonal_set=evaluate_ob_pairing_cell(B_set,T_set, double_B_set, double_T_set, CTM_cell, config_kwargs, global_args);
        print(pairing_x_set)
        print(pairing_y_set)
        print(pairing_diagonal_set)

        print('magnetization:')
        sx_set,sy_set,sz_set=evaluate_spin_cell_iPESS(B_set,T_set, double_B_set, double_T_set, CTM_cell, config_kwargs, global_args);
        print(sx_set.tolist())
        print(sy_set.tolist())
        print(sz_set.tolist())
        S2=torch.sqrt(sx_set**2+sy_set**2+sz_set**2)
        print(S2.tolist())
    elif energy_setting.model in ("triangle_spinHall", "triangle_spinfulHofstadter"):
        E_total,  ex_up_set, ey_up_set, e_diagonala_up_set, ex_dn_set, ey_dn_set, e_diagonala_dn_set, e0_set, eU_set, sx_set, sy_set, sz_set=evaluate_ob_cell_iPESS(parameters, B_set,T_set, double_B_set, double_T_set, CTM_cell, energy_setting, config_kwargs, global_args);
        print('E= '+str(E_total.item()))
        print('hopping for spin up:')
        print(ex_up_set.tolist())
        print(ey_up_set.tolist())
        print(e_diagonala_up_set.tolist())
        print('hopping for spin dn:')
        print(ex_dn_set.tolist())
        print(ey_dn_set.tolist())
        print(e_diagonala_dn_set.tolist())
        print('occupation and interaction:')
        print(e0_set.tolist())
        print(eU_set.tolist())

    

        print('magnetization components:')

        print(sx_set.tolist())
        print(sy_set.tolist())
        print(sz_set.tolist())
        print('total magnetization:')
        S2=torch.sqrt(sx_set**2+sy_set**2+sz_set**2)
        print(S2.tolist())
    elif energy_setting.model == "triangle_spinlessHofstadter":
        E_total,  ex_set, ey_set, e_diagonala_set, e0_set =evaluate_ob_cell_iPESS(parameters, B_set,T_set, double_B_set, double_T_set, CTM_cell, energy_setting, config_kwargs, global_args);
        print('E= '+str(E_total.item()))
        print('hopping:')
        print(ex_set.tolist())
        print(ey_set.tolist())
        print(e_diagonala_set.tolist())
        print('occupation:')
        print(e0_set.tolist())

    if return_ctm_err:
        return E_total,ite_err
    return E_total


def backtracking_line_search(parameters, x, direction, E0, E_min, grad, CTM_cell, D, chi,
                             ls_ctm_args, energy_setting, global_args, config_kwargs, ls):
    alpha=getattr(ls,'step0',1.0)
    shrink=getattr(ls,'alpha',3/4)
    c1=getattr(ls,'c1',1e-4)
    ls_maxiter=getattr(ls,'ls_maxiter',20)
    min_step=getattr(ls,'min_step',1e-12)
    direction,dphi0=make_descent_direction(grad,direction)
    E0=_to_float(E0)
    E_min=_to_float(E_min)
    best_x=x
    best_E=E0
    best_ite_err=numpy.inf

    with torch.no_grad():
        for ls_step in range(1,ls_maxiter+1):
            print('backtracking line search step '+str(ls_step)+', alpha='+str(alpha))
            x_trial=add_scaled_state(x,direction,alpha)
            x_trial.normalize()
            E_trial,ite_err=fx(parameters, x_trial, CTM_cell, ls_ctm_args, energy_setting, global_args, config_kwargs, return_ctm_err=True)
            E_trial_float=_to_float(E_trial)
            armijo_rhs=E0+c1*alpha*dphi0

            if E_trial_float<best_E:
                best_x=x_trial
                best_E=E_trial_float
                best_ite_err=ite_err

            if E_trial_float<=armijo_rhs:
                print('accepted alpha='+str(alpha)+', E='+str(E_trial_float))
                _save_line_search_tensor_if_strict(x_trial,E_trial_float,E_min,ite_err,D,chi,config_kwargs)
                return x_trial,E_trial_float,alpha,True,None,None

            alpha=alpha*shrink
            if alpha<min_step:
                break

    if best_E<E0:
        _save_line_search_tensor_if_strict(best_x,best_E,E_min,best_ite_err,D,chi,config_kwargs)
    print('line search failed Armijo condition; use best trial E='+str(best_E))
    return best_x,best_E,alpha,False,None,None


def _line_search_grad_eval(parameters, x, direction, alpha, AD_ctm_args, energy_setting, global_args, config_kwargs, ls=None):
    x_trial=add_scaled_state(x,direction,alpha)
    x_trial.normalize()
    grad_trial,E_trial,CTM_trial,ite_err=get_grad(parameters, x_trial, AD_ctm_args, energy_setting, global_args, config_kwargs, return_ctm_err=True)
    grad_trial=clip_state_grad(grad_trial, getattr(ls, "max_grad_norm", None))
    E_trial_float=_to_float(E_trial)
    dphi_trial=state_inner(grad_trial,direction)
    return x_trial,E_trial_float,grad_trial,dphi_trial,CTM_trial,ite_err


def _hz_accept(phi, dphi, phi0, dphi0, alpha, ls):
    delta=getattr(ls,'hz_delta',0.1)
    sigma=getattr(ls,'hz_sigma',0.9)
    epsilon=getattr(ls,'hz_epsilon',1e-6)
    wolfe=(phi<=phi0+delta*alpha*dphi0) and (dphi>=sigma*dphi0)
    approx_wolfe=(phi<=phi0+epsilon*abs(phi0)) and ((2*delta-1)*dphi0>=dphi) and (dphi>=sigma*dphi0)
    return wolfe or approx_wolfe


def print_accepted_observables(parameters, state, CTM_cell, ls_ctm_args, energy_setting, global_args, config_kwargs, ls, return_ctm_err=False):
    if getattr(ls,'print_observables',True):
        print('accepted-step observables:')
        return fx(parameters, state, CTM_cell, ls_ctm_args, energy_setting, global_args, config_kwargs, return_ctm_err=return_ctm_err)
    if return_ctm_err:
        return fx(parameters, state, CTM_cell, ls_ctm_args, energy_setting, global_args, config_kwargs, return_ctm_err=True)
    return None


def hager_zhang_line_search(parameters, x, direction, E0, E_min, grad, CTM_cell, D, chi,
                            AD_ctm_args, ls_ctm_args, energy_setting, global_args, config_kwargs, ls):
    alpha=getattr(ls,'step0',1.0)
    expand=getattr(ls,'hz_expand',2.0)
    ls_maxiter=getattr(ls,'ls_maxiter',10)
    min_step=getattr(ls,'min_step',1e-12)
    direction,dphi0=make_descent_direction(grad,direction)
    E0=_to_float(E0)
    E_min=_to_float(E_min)

    best_x=x
    best_E=E0
    best_grad=None
    best_CTM=CTM_cell
    low_alpha=0.0
    low_E=E0
    high_alpha=None

    for ls_step in range(1,ls_maxiter+1):
        print('Hager-Zhang line search step '+str(ls_step)+', alpha='+str(alpha))
        x_trial,E_trial,grad_trial,dphi_trial,CTM_trial,_ite_err=_line_search_grad_eval(
            parameters,x,direction,alpha,AD_ctm_args,energy_setting,global_args,config_kwargs,ls)

        if E_trial<best_E:
            best_x=x_trial
            best_E=E_trial
            best_grad=grad_trial
            best_CTM=CTM_trial

        if _hz_accept(E_trial,dphi_trial,E0,dphi0,alpha,ls):
            print('accepted alpha='+str(alpha)+', E='+str(E_trial)+', dphi='+str(dphi_trial))
            E_obs,ite_err_obs=print_accepted_observables(parameters, x_trial, CTM_trial, ls_ctm_args, energy_setting, global_args, config_kwargs, ls, return_ctm_err=True)
            _save_line_search_tensor_if_strict(x_trial,E_obs,E_min,ite_err_obs,D,chi,config_kwargs)
            return x_trial,E_trial,alpha,True,grad_trial,CTM_trial

        if (E_trial>E0) or (ls_step>1 and E_trial>=low_E):
            high_alpha=alpha
        elif dphi_trial>=0:
            high_alpha=alpha
        else:
            low_alpha=alpha
            low_E=E_trial
            alpha=alpha*expand
            continue

        if high_alpha is not None:
            for zoom_step in range(ls_step+1,ls_maxiter+1):
                alpha=(low_alpha+high_alpha)/2
                if alpha<min_step:
                    break
                print('Hager-Zhang zoom step '+str(zoom_step)+', alpha='+str(alpha))
                x_trial,E_trial,grad_trial,dphi_trial,CTM_trial,_ite_err=_line_search_grad_eval(
                    parameters,x,direction,alpha,AD_ctm_args,energy_setting,global_args,config_kwargs,ls)

                if E_trial<best_E:
                    best_x=x_trial
                    best_E=E_trial
                    best_grad=grad_trial
                    best_CTM=CTM_trial

                if _hz_accept(E_trial,dphi_trial,E0,dphi0,alpha,ls):
                    print('accepted alpha='+str(alpha)+', E='+str(E_trial)+', dphi='+str(dphi_trial))
                    E_obs,ite_err_obs=print_accepted_observables(parameters, x_trial, CTM_trial, ls_ctm_args, energy_setting, global_args, config_kwargs, ls, return_ctm_err=True)
                    _save_line_search_tensor_if_strict(x_trial,E_obs,E_min,ite_err_obs,D,chi,config_kwargs)
                    return x_trial,E_trial,alpha,True,grad_trial,CTM_trial

                if (E_trial>E0) or (E_trial>=low_E):
                    high_alpha=alpha
                elif dphi_trial>=0:
                    high_alpha=alpha
                else:
                    low_alpha=alpha
                    low_E=E_trial
            break

        alpha=alpha*getattr(ls,'alpha',0.5)
        if alpha<min_step:
            break

    if best_E<E0:
        E_obs,ite_err_obs=print_accepted_observables(parameters, best_x, best_CTM, ls_ctm_args, energy_setting, global_args, config_kwargs, ls, return_ctm_err=True)
        _save_line_search_tensor_if_strict(best_x,E_obs,E_min,ite_err_obs,D,chi,config_kwargs)
    print('Hager-Zhang line search failed; use best trial E='+str(best_E))
    return best_x,best_E,alpha,False,best_grad,best_CTM


def line_search(parameters, x, direction, E0, E_min, grad, CTM_cell, D, chi,
                AD_ctm_args, ls_ctm_args, energy_setting, global_args, config_kwargs, ls):
    method=getattr(ls,'line_search','hager_zhang').lower()
    if method in ('hager_zhang','hager-zhang','hz'):
        return hager_zhang_line_search(parameters,x,direction,E0,E_min,grad,CTM_cell,D,chi,
                                       AD_ctm_args,ls_ctm_args,energy_setting,global_args,config_kwargs,ls)
    elif method in ('backtracking','armijo'):
        return backtracking_line_search(parameters,x,direction,E0,E_min,grad,CTM_cell,D,chi,
                                        ls_ctm_args,energy_setting,global_args,config_kwargs,ls)
    else:
        raise ValueError("unknown line search method: "+str(method))


def nonlinear_cg_opt(parameters, D,chi, x0, AD_ctm_args, ls_ctm_args, energy_setting, global_args, config_kwargs, ls):
    print("nonlinear conjugate gradient optimization")
    print("D="+str(D))
    print("chi="+str(chi))
    x=x0.copy()
    E_min=numpy.inf
    grad_prev=None
    direction_prev=None
    gnorm=100
    iter=1
    beta_rule=getattr(ls,'cg_beta','PRP')
    grad=None
    E_float=None
    CTM_cell=None

    while (iter<=ls.maxiter) and (gnorm>ls.gtol):
        start_time=time.time()
        print("optim iteration "+str(iter))
        x.normalize()
        if grad is None:
            grad,E,CTM_cell=get_grad(parameters, x, AD_ctm_args, energy_setting, global_args, config_kwargs)
            grad=clip_state_grad(grad, getattr(ls, "max_grad_norm", None))
            E_float=_to_float(E)
            gnorm=grad.norm()

        if E_float<E_min:
            E_min=E_float

        if grad_prev is None:
            direction=scale_state(grad,-1.0)
        else:
            if beta_rule.upper()=="FR":
                beta=state_inner(grad,grad)/max(state_inner(grad_prev,grad_prev),1e-30)
            else:
                y=subtract_state(grad,grad_prev)
                beta=state_inner(grad,y)/max(state_inner(grad_prev,grad_prev),1e-30)
                beta=max(0.0,beta)
            direction=state_axpy(scale_state(grad,-1.0),direction_prev,1.0,beta)
            print("CG beta="+str(beta))

        x_new,E_new,alpha,accepted,grad_new,CTM_new=line_search(
            parameters,x,direction,E_float,E_min,grad,CTM_cell,D,chi,
            AD_ctm_args,ls_ctm_args,energy_setting,global_args,config_kwargs,ls)

        end_=time.time()
        print('time consumed: '+time.strftime("%H hours, %M minuts, %S seconds", time.gmtime(end_ - start_time)))

        grad_prev=grad
        direction_prev=direction
        x=x_new
        if E_new<E_min:
            E_min=E_new
        if not accepted:
            direction_prev=None
            grad_prev=None
        if grad_new is not None:
            grad=grad_new
            E_float=E_new
            CTM_cell=CTM_new
            gnorm=grad_new.norm()
        else:
            grad=None
            E_float=None
            CTM_cell=None
        iter+=1

    return x


def lbfgs_direction(grad, history):
    if len(history)==0:
        return scale_state(grad,-1.0)

    q=grad.copy()
    alpha_list=[]
    for s,y,rho in reversed(history):
        alpha=rho*state_inner(s,q)
        alpha_list.append(alpha)
        q=subtract_state(q,scale_state(y,alpha))

    s_last,y_last,rho_last=history[-1]
    yy=state_inner(y_last,y_last)
    gamma=state_inner(s_last,y_last)/yy if yy>1e-30 else 1.0
    r=scale_state(q,gamma)

    for (s,y,rho),alpha in zip(history,reversed(alpha_list)):
        beta=rho*state_inner(y,r)
        r=state_axpy(r,s,1.0,alpha-beta)

    return scale_state(r,-1.0)


def lbfgs_opt(parameters, D,chi, x0, AD_ctm_args, ls_ctm_args, energy_setting, global_args, config_kwargs, ls):
    print("L-BFGS optimization")
    print("D="+str(D))
    print("chi="+str(chi))
    x=x0.copy()
    history=[]
    history_size=getattr(ls,'history_size',8)
    E_min=numpy.inf
    gnorm=100
    iter=1
    grad=None
    E_float=None
    CTM_cell=None

    while (iter<=ls.maxiter) and (gnorm>ls.gtol):
        start_time=time.time()
        print("optim iteration "+str(iter))
        x.normalize()
        if grad is None:
            grad,E,CTM_cell=get_grad(parameters, x, AD_ctm_args, energy_setting, global_args, config_kwargs)
            grad=clip_state_grad(grad, getattr(ls, "max_grad_norm", None))
            E_float=_to_float(E)
            gnorm=grad.norm()
        if E_float<E_min:
            E_min=E_float

        direction=lbfgs_direction(grad,history)
        x_new,E_new,alpha,accepted,grad_new,CTM_new=line_search(
            parameters,x,direction,E_float,E_min,grad,CTM_cell,D,chi,
            AD_ctm_args,ls_ctm_args,energy_setting,global_args,config_kwargs,ls)

        if accepted:
            if grad_new is None:
                grad_new,E_grad_new,CTM_new=get_grad(parameters, x_new, AD_ctm_args, energy_setting, global_args, config_kwargs)
                grad_new=clip_state_grad(grad_new, getattr(ls, "max_grad_norm", None))
                E_new=_to_float(E_grad_new)
            s=subtract_state(x_new,x)
            y=subtract_state(grad_new,grad)
            sy=state_inner(s,y)
            if sy>1e-30:
                history.append((s,y,1.0/sy))
                if len(history)>history_size:
                    history.pop(0)
            else:
                print("skip L-BFGS history update because s dot y is too small")
            grad=grad_new
            E_float=E_new
            CTM_cell=CTM_new
            gnorm=grad_new.norm()
        else:
            history=[]
            if grad_new is not None:
                grad=grad_new
                E_float=E_new
                CTM_cell=CTM_new
                gnorm=grad_new.norm()
            else:
                grad=None
                E_float=None
                CTM_cell=None

        x=x_new
        if E_new<E_min:
            E_min=E_new
        end_=time.time()
        print('time consumed: '+time.strftime("%H hours, %M minuts, %S seconds", time.gmtime(end_ - start_time)))
        iter+=1

    return x


def optimize_iPESS(parameters, D,chi, x0, AD_ctm_args, ls_ctm_args, energy_setting, global_args, config_kwargs, ls):
    method=getattr(ls,'method','stochastic').lower()
    if method in ('stochastic','sgd','random'):
        return stochastic_opt(parameters,D,chi,x0,AD_ctm_args,ls_ctm_args,energy_setting,global_args,config_kwargs,ls)
    elif method in ('cg','conjugate_gradient','conjugate-gradient'):
        return nonlinear_cg_opt(parameters,D,chi,x0,AD_ctm_args,ls_ctm_args,energy_setting,global_args,config_kwargs,ls)
    elif method in ('lbfgs','l-bfgs','lgfbs'):
        return lbfgs_opt(parameters,D,chi,x0,AD_ctm_args,ls_ctm_args,energy_setting,global_args,config_kwargs,ls)
    else:
        raise ValueError("unknown optimization method: "+str(method))



def stochastic_opt(parameters, D,chi, x0, AD_ctm_args, ls_ctm_args, energy_setting, global_args, config_kwargs, ls):

    print("stochastic optimization")
    print("D="+str(D));
    print("chi="+str(chi));
    x = x0.copy();


    iter = 1
    gnorm=100;
    E_min=100;
    delta=getattr(ls,'delta0',1e-3);
    while (iter < ls.maxiter) & (gnorm > ls.gtol) & (delta>1e-6):
        start_time = time.time()

        print("optim iteration "+str(iter))
        x.normalize();
        state_grad,E_grad, CTM_cell=get_grad(parameters, x, AD_ctm_args, energy_setting, global_args, config_kwargs);
        state_grad=clip_state_grad(state_grad, getattr(ls, "max_grad_norm", None))
        
        if iter==1:
            E_min=E_grad.item();
        with torch.no_grad():
            ls_step=1;
            E_updated=100;
            while E_updated>E_min:
                print('line search step '+str(ls_step))
                xgrad_rand=get_random_grad(state_grad,delta)
                print("norm of random grad:"+str(xgrad_rand.norm()))
                x_updated=subtract_state(x,xgrad_rand)

                E_updated,ite_err=fx(parameters, x_updated, CTM_cell, ls_ctm_args, energy_setting, global_args, config_kwargs, return_ctm_err=True);
                E_updated=E_updated.item();
                ls_step=ls_step+1;
                
                if (E_updated<E_min) :
                    history_min_E=E_min
                    E_min=E_updated
                    _save_line_search_tensor_if_strict(x_updated,E_updated,history_min_E,ite_err,D,chi,config_kwargs)
                    end_ = time.time()
                    print('time consumed: '+time.strftime("%H hours, %M minuts, %S seconds", time.gmtime(end_ - start_time)))
                    break;
                else:
                    delta=delta*ls.alpha;
                    print('new delta: '+str(delta))
            x=x_updated;
            x.normalize()
            iter += 1
            gnorm = state_grad.norm();
    
    return x
