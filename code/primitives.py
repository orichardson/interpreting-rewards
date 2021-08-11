# import numpy as np
# import autograd.numpy as np
# from autograd import isinstance
import torch
import math
# from torch import tensor
tensor = torch.tensor
from torch import logsumexp 
# from scipy.special import logsumexp

# import pandas as pd
# import gym

from utils import matches_any
import tracing
# from tracing import TraceStore
from collections import defaultdict
from environs import Env


# if store_iters != None:
    # for kind, names in valid_names.items():
        # if matches_any(store_iters, *names):
            # trace[kind].append(kw_data[kind])

# kd = dict(keepdims=True) # the numpy version
kd = dict(keepdim=True) # the torch version


def t_max(arr, temp = 0, axis=None):
    if temp == 0:
        # @autograd fix
        # return np.asarray(arr).max(axis=axis, **kd)
        return (arr).max(axis=axis, **kd).values
    else:
        ### Would like to do this, but numerically unstable. 
        # return temp * np.log( np.exp(arr/temp).sum(**ufunc_kwargs) )
        ### Unfortunately scipy's logsumexp deals with temperature wrong.
        return temp*logsumexp(torch.div(arr, temp), axis=axis, **kd)
        
    
def t_argmax(arr, temp=0, axis=None):
    if temp == 0:
        if not torch.is_tensor(arr):
            arr = torch.tensor(arr)
        almost = (arr == arr.max(axis=axis, **kd).values) +0.0
        return almost / almost.sum(axis=axis, **kd)

    else:
        ascaled = torch.div(arr, temp)
        almost = torch.exp(ascaled - logsumexp(ascaled, axis=axis, **kd))
        return almost / almost.sum(axis=axis, **kd)
    
    # (s,a) =>  E_{s'} - log T(s' | a,s)
    # return (-np.log(env.TT)*env.TT).sum(axis=2)
    # raise NotImplementedError()


# def optimal_policy( r, ENV ):
# def _preprocess(env, R, pi)
    
def value_iter(env, R, γ, iters=100, init=None, temperature=0, trace=tracing.none): 
    if not torch.is_tensor(γ):
        γ = torch.tensor([γ])
    
    V = torch.zeros(env.Vshape) if init is None else init
    if trace: trace(V=V)
    for it in range(iters):
        # V = max_a \Ex_{s'~p|a,s} [ r(a,s) + \gamma V(s') ]
        # Qs =  (env.TT * (R + γ * V.T)).sum(axis=2,**kd)
        
        # print(t_max( Q(env, R, V, γ) - V, temperature, axis=1))
        
        V = t_max( Q(env, R, V, γ) - V, temperature, axis=1) + V
        # constant shift should not change softmax...
        # V = t_max( Q(env, R, V, γ), temperature, axis=1) 

        if trace: trace(V= V *  (1 - γ) / (1 - torch.pow(γ, it+1)))
            # )

        
    return V  *  (1 - γ) / (1 - torch.pow(γ, iters))

def value_iter_recencybias(env, R, γ, iters=100, init=None, temperature=0): 
    """ 
    Renormalizes every timestep instead of  
    """
    V = torch.zeros(env.Vshape) if init is None else init
    for it in range(iters):
        # V = (env.TT * (R * (1-γ) + γ * V.T)).sum(axis=2,**kd).max(axis=1,**kd)
        Qs = (env.TT * (R * (1-γ) + γ * V.T)).sum(axis=2,**kd)
        V = t_max( Qs, temperature, axis=1)

    return V

def policy_eval(env:Env, pi, R, γ, iters=100, init=None, trace=tracing.none):
    if not torch.is_tensor(γ):
        γ = torch.tensor([γ])
    # pi : (S, A)
    # print("in 'policy_eval': \t  π.shape", np.shape(pi))
    V = (torch.zeros(env.Vshape) if init is None else init).reshape(env.Vshape)
    
    # pi_arr = torch.array(pi).reshape(env.πshape)
    pi_arr = pi.reshape(env.πshape)
    if trace: trace(V=V)    
    
    for it in range(iters):
        # _____ RECENCY BIAS _____
        # V = (pi_arr * env.TT * (R + γ * V.T)).sum(axis=(1,2)) / (1 + γ)
        #!! WORRY: (1+gamma) is not the usual (1-\gamma)...
        
        # THE USUAL VERSION
        V = (pi_arr * env.TT * (R + γ * V.T)).sum(axis=(1,2), **kd) 
        if trace: trace(V=V *  (1 - γ) / (1 - torch.pow(γ, it+1))
            )
        
    # return V
    return V *  (1 - γ) / (1 - torch.pow(γ, iters))
    
def policy_eval_recencybias(env:Env, pi, R, γ, iters=100, init=None):
    """
    A value function is often re-normalized by dividing through
    by sum of the geometric series decaying at rate gamma for the infinite
    horizon case (or not at all). This renormalization is gives the expected
    future rate of acruing value from this point onwards.
    
    Here, γ \in [0,1] is instead interpreted as the probability that what 
    you will get value from future, as opposed to the present. 
    
    
    It is the "dual" of the standard version, which has an initiality bias,
    rather than a recency bias. That makes it more amenable to backwards
    computation.
    
    
    Parameters
    ----------
    env : Env.
    pi : (S, A, 1)
        The policy pi(S | A)
    R : (S?1, A?1, S?1) 
        Reward tensor, that broadcasts to be the same shape as environ tensor. 
    γ : float
        Discount factor, with recency bias. See discussion above.
    iters : TYPE, optional
        number of iterations. The default is 100.
    init : array (S, A, 1), optional
        Initial Value for V. The default is None.

    Returns
    -------
    V : (S, A, 1)
        The returned value function

    """
    V = torch.zeros(env.Vshape) if init is None else init
    # pi_arr = np.array(pi).reshape(env.πshape)
    pi_arr = pi.reshape(env.πshape)
    
    for it in range(iters):
        V = (pi_arr * env.TT * ((1 - γ) * R + γ * V.T)).sum(axis=(1,2))        
        
    return V


def Q(env, R, V, γ):
    # print("in Q:  \t V.shape: ", np.shape(V))
    # Vnext = tensor(V).reshape(1,1,env.nS)
    Vnext = V.reshape(1,1,env.nS)
    return (env.TT*(R + γ * Vnext)).sum(axis=2,**kd) 
    # / (1 + γ)
    
def Adv(env, R, γ, V=None):
    if V is None:
        V = value_iter(env, R, γ)
    return Q(env, R, V, γ) - V
    
def best_policy( env, R, V, γ, temperature=0):
    # print("in 'best policy'.  \t V.shape: ", np.shape(V), "R.shape:  ", np.shape(R))
    assert V.shape == env.Vshape
    return t_argmax(Q(env, R, V, γ) - V, temperature, axis=1)
    
    # Qmaxima = (Qvals == Qvals.max(axis=1,**kd))+0.0 if temperature == 0  \
    #     else np.exp( Qvals / temperature )        
    # return Qmaxima / (Qmaxima.sum(axis=1,**kd))

    
## DEPRICATED ##
def event_joint_iter(env, init=None, iters=100):
    dist = torch.ones(env.SAshape) if init is None \
        else init.reshape(env.SAshape)
    dist /= dist.sum()
    
    # iterative fixed pt computation (i.e., right eigenvector)
    for it in range(iters):
        dist = (env.TT * dist).sum(axis=0,**kd).transpose(0,2)
    
    return dist


def visitation_iter(env, pi, init=None, iters=100):
    dist = torch.ones(env.Sshape) / env.nS if init is None \
        else init.reshape(env.Sshape)
    pi = pi.reshape(env.pi_shape)
    
    # iterative fixed pt computation (i.e., right eigenvector)
    for it in range(iters):
        dist = (env.TT * dist * pi).sum(axis=(0,1), **kd).transpose(0,2)
        dist /= dist.sum() # just a precaution...
    
    return dist            

    
def fwd(env, R, γ, 
        policy_improve_iters=0, 
        val_iters=100,
        alternations=1,
        temp=0, trace=tracing.none):
    V = None
    for j in range(alternations):
        V = value_iter(env, R, γ, val_iters, init=V, temperature=temp, trace=trace+str(j)+'v')
        π = best_policy(env, R, V, γ, temperature=temp)
        # if trace: trace(V=V, π=π)
        # if trace: trace(V0=V, π0=π)
        
        for i in range(policy_improve_iters):
            V = policy_eval(env, π, R, γ, iters=val_iters, init=V, trace=trace+str(j)+'e'+str(i))
            π = best_policy(env, R, V, γ, temperature=temp)
            # if trace: trace(V=V, π=π)

    return π    
    
    
def MCE_IRL(env:Env, pi, γ, lr=1, lr_decay=0.9, 
            policy_improve_iters=0,
            value_iters = 100,
            visit_iters = 100,
            temperature = 0,
            iters=100):
    pi = pi.reshape(env.SA_shape)
    R = torch.zeros(env.Tshape)
    D_want = visitation_iter(env, pi, iters=visit_iters)

    
    for it in range(iters):
        pi_R = fwd(env, R, γ, policy_improve_iters, value_iters, temp=temperature)
        D_R = visitation_iter(env, pi_R, iters=visit_iters)
        R += lr * ( D_want * pi - D_R * pi_R)
        lr *= lr_decay
        
    return R
    
def torch_IRL(env, pi, γ):
    pass


def regularized_IRL(env:Env, pi, γ,
            regularizer,
            lr=1, lr_decay=0.9, 
            policy_improve_iters=0,
            value_iters = 100,
            visit_iters = 100,
            temperature = 0,
            iters=100):
    pi = pi.reshape(env.SA_shape)
    R = torch.zeros(env.Tshape)
    D_want = visitation_iter(env, pi, iters=visit_iters)

    
    for it in range(iters):
        pi_R = fwd(env, R, γ, policy_improve_iters, value_iters, temp=temperature)
        D_R = visitation_iter(env, pi_R, iters=visit_iters)
        R += lr * ( D_want * pi - D_R * pi_R)
        lr *= lr_decay
        
    return R

    
######################################################    
#           UNDER CONSTRUCTION 
######################################################
class Reward:
    def __init__(self, arr=None):
        self.R = tensor(arr)


    # ######### MEASURES OF GOAL-DIRECTEDNESS ########
    # def value_variance(self, D, V):
    #     """
    #     Notes: 
    #      * Depends heavily on this distribution.  
    #     """
    #     assert np.allclose(np.sum(D, axis=0),1),\
    #         "0-axis sum isn't one, but %.3f" % np.sum(D, axis=0)
    #     # assert D.shape[0] == R.shape[0]
    # 
    #     mean = (D * V).sum()
    #     return ((D*V - mean)**2).sum()
    # 
    def diff(self, env, γ):
        # really want ∂/∂γ softmax(Q(A,S));
        # a cheap approximation is softmax(V(S' | gamma=1))-softmax(R(A,S))

        R = self.R
        ### ??? what do we do with gamma?
        V = value_iter(env, R, γ, temperature=0.001)
        # V_sprime = (env.TT * (V)).sum(axis=2)
        
        
        # Cosine Similarity
        softmax_Adv = t_argmax(Adv(env,R,γ,V), temp=0.001, axis=1)
        softmax_R = t_argmax(R+torch.zeros(env.SAshape), temp=0.001, axis=1)
        return (softmax_Adv * softmax_R).sum() / (softmax_Adv.sum() * softmax_R.sum() )
        
        
        
    def canonicalize(self, DA=None, DS=None):
        # R +  Ex_[ gamma R(s', A, S') - R(s, A, S') + gamma R(S,A,S') ]
        R = self.R
        if DA is None: DA = torch.ones()
        DA = tensor(DA) / torch.sum(DA, axis=(0,1))
        if DS is None: DS = torch.ones()
        DS = tensor(DS) / torch.sum(DS, axis=(0,1))
        
        raise NotImplementedError()
    
    
# def logZ(R, temp):
#     return temp * np.log(np.exp(R / temp).sum(axis=1))
def policyReward(pi,  temp=0):
    import numpy as np
    with np.errstate(divide='ignore'):
        infocost = np.log(pi)
        
    am = t_argmax(infocost, temp, axis=1)
    val = t_max(pi, temp, axis=1)
    # val = am.max(axis=1, **kd)
    return am - val
    
def adjust_both(env:Env, π, ρ, γ, 
        # balance = 0.5, 
        iters=100, visit_iters=100, value_iters=100,
        lr = 1, lr_decay=0.95, temp=0,
        store_iters = None):
    """ 
    """
    valid_names = dict(
        D = ["D", "visit"], 
        V = ["V", "value"],
        π = ["π", "pi", "policy"],
        ρ = ["ρ", "R", "reward"])
    trace = defaultdict(list)

    # π_new = np.array(pi).reshape(env.SA_shape)
    # ρ_new = np.array(ρ) 
    # np.zeros(env.Tshape)
    # D_want = visitation_iter(env, pi, iters=visit_iters)

    def maybe_store(**kw_data):
        if store_iters != None:
            for kind, names in valid_names.items():
                if matches_any(store_iters, *names):
                    trace[kind].append(kw_data[kind])
    
    # V = None
    V = value_iter(env, ρ, γ, iters=value_iters, temperature=temp)
    D = visitation_iter(env, π, iters=visit_iters)
        
    for it in range(iters):
        V_new = policy_eval(env, π, ρ, γ, iters=value_iters, init=V)

        # update reward based on old policy, new state distribution
        # ρ_new = ρ + lr * ( D_new * π_new - D * π)
        # ρ_new = ρ + lr * ( π * (D_new - D) )
        # ρ_new = ρ + lr * ( (π_new - π) * D )
        # PROBLEM: π does not give rise to D; that was the last policy. 
        # TODO: this needs a logarithm somewhere?
        # maybe like this, except it doesn't use the policy...
        # ρ_new = ρ + lr * np.log( D_new / D)
        
        # update new policy based on old rewards, new values?
            # > seems problematic; rewards are prior to values.
        # π_new = best_policy(env, ρ, V_new, γ, temperature=temp)

        # update policy based on new values. 
        π_new = best_policy(env, ρ, V_new, γ, temperature=temp)
        
        
        D_new = visitation_iter(env, π_new, iters=visit_iters)
        # ρ_new = ρ + lr * ( (π_new - π) * D )
        ρ_new = ρ # + lr * ( D_new * π_new - D * π)

        
        #update
        D = D_new
        V = V_new
        π = π_new
        ρ = ρ_new
        lr *= lr_decay
        maybe_store(D=D_new, V=V_new, π=π_new, ρ=ρ_new)
        
    return (π, ρ) + ((trace,) if store_iters != None else ())   
    
