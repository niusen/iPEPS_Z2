import random
import yastn
import numpy,torch
from ansatz.triangle_iPESS import *
from ctmrg.Fermionic_CTMRG_unitcell_iPESS import *
from model.fermion_ob_iPESS import *
from config.settings import *
from config.config import *
import time



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

def cost_fun(parameters,state, ctm_args, energy_setting, global_args, config_kwargs):
    state.require_grad(True)
    B_set=state.B_set
    T_set=state.T_set
    init=INITCTMARGS()
    CTM0=None;
    CTM_cell, double_B_set,double_T_set,ite_num,ite_err=Fermionic_CTMRG_cell_iPESS(B_set,T_set,init,CTM0, ctm_args, global_args);

    E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell_iPESS(parameters, B_set,T_set, double_B_set, double_T_set, CTM_cell, energy_setting, config_kwargs, global_args);
    # print(E_total)
    # print(ex_set)
    # print(ey_set)
    # print(e_diagonala_set)
    # print(e0_set)
    # print(eU_set)
    print('E='+str(E_total.item()))
    return E_total,CTM_cell
def get_grad(parameters,state, ctm_args, energy_setting, global_args, config_kwargs):
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
    print("norm of grad:"+str(state_grad.norm()));
    return state_grad,E, CTM_cell

    
def subtract_state(state1,state2):
    #state1-state2*coe
    B_set_new=OrderedDict()
    T_set_new=OrderedDict()
    for key in state1.B_set:
        B_set_new.update({key: state1.B_set[key]-state2.B_set[key]})
        T_set_new.update({key: state1.T_set[key]-state2.T_set[key]})
    state_new=IPESS_TRIANGLE(B_set_new, T_set_new, state1.global_args);
    return state_new



def fx(parameters,state, CTM0, ls_ctm_args, energy_setting, global_args, config_kwargs):
    state.require_grad(False)
    B_set=state.B_set
    T_set=state.T_set
    init=INITCTMARGS()
    init.reconstruct_CTM=False;
    
    CTM_cell, double_B_set,double_T_set,ite_num,ite_err=Fermionic_CTMRG_cell_iPESS(B_set,T_set,init,CTM0, ls_ctm_args, global_args);

    E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell_iPESS(parameters, B_set,T_set, double_B_set, double_T_set, CTM_cell, energy_setting, config_kwargs, global_args);
    print('E= '+str(E_total.item()))
    print(ex_set.tolist())
    print(ey_set.tolist())
    print(e_diagonala_set.tolist())
    print(e0_set.tolist())
    print(eU_set.tolist())

    print('magnetization:')
    sx_set,sy_set,sz_set=evaluate_spin_cell_iPESS(B_set,T_set, double_B_set, double_T_set, CTM_cell, config_kwargs, global_args);
    print(sx_set.tolist())
    print(sy_set.tolist())
    print(sz_set.tolist())
    S2=torch.sqrt(sx_set**2+sy_set**2+sz_set**2)
    print(S2.tolist())
    return E_total



def stochastic_opt(parameters, D,chi, x0, AD_ctm_args, ls_ctm_args, energy_setting, global_args, config_kwargs, ls):

    print("stochastic optimization")
    print("D="+str(D));
    print("chi="+str(chi));
    x = x0.copy();


    iter = 1
    gnorm=100;
    E_min=100;
    delta=ls.delta0;
    while (iter < ls.maxiter) & (gnorm > ls.gtol) & (delta>1e-6):
        start_time = time.time()

        print("optim iteration "+str(iter))
        x.normalize();
        state_grad,E_grad, CTM_cell=get_grad(parameters, x, AD_ctm_args, energy_setting, global_args, config_kwargs);
        
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

                E_updated=fx(parameters, x_updated, CTM_cell, ls_ctm_args, energy_setting, global_args, config_kwargs);
                E_updated=E_updated.item();
                ls_step=ls_step+1;
                
                filenm='Z2_D'+str(D)+'_chi'+str(chi);
                if (E_updated<E_min) :
                    E_min=E_updated
                    save_triangle_iPESS(x_updated.B_set, x_updated.T_set, filenm, config_kwargs)
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











