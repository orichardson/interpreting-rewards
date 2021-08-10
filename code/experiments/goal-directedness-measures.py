
%cd code
%load_ext autoreload
%autoreload 2

#%%

import numpy as np
import primitives as P
Reward = P.Reward
from environs import Env, GridWorld
from utils import window
from store import TensorLibrary, fz
from dist import CPT
import seaborn as sns


#### Environments ####
environments = TensorLibrary()

##>> grid world
E = GridWorld.make(6,8,0.1)
S = E.S
A = E.A

def decode(T):
    S,A = decode._S, decode._A
    return Env(S,A, CPT.make_stoch(S&A, +S, T.reshape(len(S)*len(A), len(S))))

decode._S = S
decode._A = A


environments.ushape = (len(S), len(A), len(S))
environments.decoder = decode # probably same problem but we'll see
    # lambda T:  Env(
    # S,A, CPT.make_stoch(S&A, +S, T.reshape(len(S)*len(A), len(S))))
environments("grid world").set(E.TT)

##>> long line (left-right, book position) with 
#    UP= random teleport OR stay, down = reset.
T = np.zeros(environments.ushape)
statesByXthenY = sorted(enumerate(S), 
    key=(lambda ixy: (ixy[1][1], ixy[1][0])))

#the first state needs a left identity
i0, xy_0 = statesByXthenY[0]
T[i0, A.idx("left"), i0] = 1
#and the last state needs  a right one
il, xy_l = statesByXthenY[-1]
T[il, A.idx("right"), il] = 1


for ((i_prev,xy_prev), (i,xy)) in window(statesByXthenY,2):
    T[i_prev, A.idx("right"), i] = 1
    T[i, A.idx("left"), i_prev] = 1

T[:, A.idx("up"), :] = 1 / len(S)
T[:, A.idx("down"), 0] = 1


environments("long line").set(T)
# E_long_line = Env(S, A, CPT.make_stoch(S&A, +S, T.reshape(len(S)*len(A), len(S))))

##>> irreversible/ difficult to reverse action: likely not to move left. 
for pLeft in [0.5, 0.1, 0.01, 0.001]:
    T = np.array(E.TT)
    T[:, A.idx("left"), :] *= pLeft

    for (si, s) in enumerate(S.ordered):
        T[si, A.idx("left"), si] += 1-pLeft
        
    environments("no left", p_left=pLeft).set(T)


##>> inaccessible states + permanent choice
environments("random", det=False).set( Env.generate(A,S,spec="random") )
environments("random", det=True).set( Env.generate(A,S,spec="random.det") )

environments.tagAll("base")
# For each environment:
##>> variants with +ϵ smoothing, temperature adjustment
for S, τ in dict(environments.tensordata).items():
    for ϵ in [0.01]:
        for p in [1]:
            τnew = np.power(τ +  ϵ, p)
            τnew = τnew / τnew.sum(axis=1, keepdims=True)
            environments(*(S-{'base'}))(smoothing=ϵ,power=p).set( τnew )



# the code for deleting a bad environment
# for S, τ in dict(environments.tensordata).items():
#     if any(isinstance(s,frozenset) for s in S) 
#         del environments.tensordata[S]
#
#%%
#### Rewards ####
rewards = TensorLibrary(shape=None)

# single reward point (3,3) and disreward point (1,1)
R_dipole = np.zeros((6,8))
R_dipole[3,3] = 1
R_dipole[1,1] = -1
R_dipole = R_dipole.reshape(E.Sshape)
rewards("dipole").set(R_dipole)

# lava
R_lava = np.zeros((6,8))
R_lava[0,:] = -5
R_lava[-1,:] = -5
R_lava[:,0] = -5
R_lava[:,-1] = -5
R_lava = R_lava.reshape(E.Sshape)
rewards("lava").set(R_lava)

# random behavior (myopic)
# π_rand = E.random_policy(det=True)
rewards("match policy").set(E.random_policy(det=True))

if False: # don't bother with this yet.
    # advantage, for base rewards
    for KR in rewards.filter(lambda kr : len(kr) == 1).matches:
        for KE in environments("base").matches:
            for γ in [0.5, 0.9, 0.99]:
                rewards("advantage", env=KE-{'base'}, r0=KR, advγ=γ).set(
                    P.Adv(+environments(*KE), +rewards(*KR), γ) )

    #novelty, for base rewards
    for K in environments("base").matches:
        rewards("novelty", env=K).set( (+environments(*K)).novelty )


#%%
####
V = P.value_iter(E, R_dipole, 0.9)
# P.value_iter(E, R_dipole, 0.5, iters=100)
# P.Q(E, R_dipole, np.zeros(E.Vshape), 0.9)

π1 = P.fwd(E, R_dipole, 0.99)
π2 = P.fwd(E, P.Adv(E, R_dipole, 0.99), 0.99)
assert np.allclose(π1,π2)

##### Metrics ####
# of the form   Env, Reward, γ => ℝ
def value_variance_metric(E, R, γ):
    V = P.value_iter(E,R,γ)
    π = P.best_policy(E,R,V,γ)
    D = P.visitation_iter(E,π)
    
    mean = (D * V).sum()
    return (D*(V - mean)**2).mean()

def diff_metric(E, R, γ):
    V = P.value_iter(E, R, γ, temperature=0.001)
    
    # Cosine Similarity
    softmax_Adv = P.t_argmax(P.Adv(E,R,γ,V), temp=0.001, axis=1)
    softmax_R = P.t_argmax(R+np.zeros(E.SAshape), temp=0.001, axis=1)
    return (softmax_Adv * softmax_R).sum() / \
            np.sqrt((softmax_Adv.sum() * softmax_R.sum() ))
    # return Reward(R).diff(E, γ)
    
    
def diff_metric2(E, R, γ, ratio):
    V = P.value_iter(E, R, γ, temperature=0.001)
    
    # Cosine Similarity
    γsmall = γ * ratio
    γbig = 1 + (γ - 1)*ratio
    softmax1 = P.t_argmax(P.Adv(E,R,γsmall,V), temp=0.001, axis=1)
    softmax2 = P.t_argmax(P.Adv(E,R,γbig,V), temp=0.001, axis=1)
    return (softmax1 * softmax2).sum() / \
            np.sqrt((softmax1.sum() * softmax2.sum() ))
    # return Reward(R).diff(E, γ)
    
def expected_softmaxdiff(E, R, γ):
    V = P.value_iter(E, R, γ, temperature=0.001)
    π = P.best_policy(E,R,V,γ)
    D = P.visitation_iter(E,π)
    
    # Cosine Similarity
    γsmall = γ * ratio
    γbig = 1 + (γ - 1)*ratio
    softmax1 = P.t_argmax(P.Adv(E,R,γsmall,V), temp=0.001, axis=1)
    softmax2 = P.t_argmax(P.Adv(E,R,γbig,V), temp=0.001, axis=1)
    
    expsoftmax1 = (D*softmax1).sum(axis=0)
    expsoftmax2 = (D*softmax2).sum(axis=0)
    # weight by 
    return (expsoftmax1 * expsoftmax2).sum() / \
            np.sqrt( expsoftmax1.sum() * expsoftmax2.sum() )
    
    
from functools import partial

metrics = {
    fz("valvar") : value_variance_metric,
    # fz("diff") : diff_metric, 
    fz("E_cossim") : expected_softmaxdiff,
    **{  fz("cossim", ratio=r) : partial(diff_metric2, ratio=r) 
            for r in [0.1, 0.5, 0.9]  }
}

# P.t_argmax(R_dipole+np.zeros(E.SAshape), temp=0.001,axis=1)
#%%
RESULTS = TensorLibrary()
for KE, E in environments("base"):
    print("ENVIRONMENT", KE)
    for KR, R in rewards.without("novelty"):
        print("REWARD: ", KR)
        for KM, M in metrics.items():
            for γ in [0.5, 0.9, 0.99]:
                print(KE,KR,KM,γ,'|  \t  ', M(E,R,γ))
                RESULTS(R=KR, E=KE, M=KM, γ=γ).set( M(E,R,γ) )

import pickle
with open('tensordata.%d.pickledict'%len(RESULTS), 'wb') as f:
    print("wrote to ", f.name)
    pickle.dump(RESULTS.tensordata, f)



# with open('tensordata.9000.pickledict', 'rb') as f:
#     BIG_RESULTS = TensorLibrary()
#     BIG_RESULTS.tensordata = pickle.load(f)

[*RESULTS.values_for_key('M')]


RESULTS.df.stack()

RES0 = RESULTS(M=fz('diff2',ratio=0.1))
# RES0 = RESULTS(M=('valvar',))

# from IPython.core.display import display, HTML
# display(HTML(__.to_html()))

RESULTS(M=fz('diff2', ratio=0.1), γ=0.5).dataframe_by_attrs('R', 'E')

sns.heatmap(RESULTS(M=fz('diff2', ratio=0.1), γ=0.5).dataframe_by_attrs('R', 'E'),
    annot=True, fmt=".3f")
# 

# D.style.applymap(lambda v: 'background-color:rgba(255,0,0,%f);'% v)


RESULTS(M=fz('diff2', ratio=0.1)).dataframe_by_attrs('R', 'γ')

RES0.dataframe_by_attrs('γ', 'E')
# BIG_RESULTS(M=('diff2', ('ratio', 0.1))).dataframe_by_attrs('R','E')

import pandas as pd
dod = RESULTS.dataframe_by_attrs('R', 'M')
dod2 = { k1 : { k2 : v for k2,v in dod[k1].items()} for k1 in dod}
dod2

pd.DataFrame({ 1 : {frozenset(('a', 2, 'q')) : 0, frozenset(('b', 1)) : 1, frozenset(('a', 2, 3)): 3}})


pd.DataFrame(dod2)

import itertools
RESULTS.dataframe_by_attrs('R', 'M')
RESULTS.dataframe_by_attrs('M', 'R')


RESULTS.dataframe_by_attrs("M", "R")


from matplotlib import pyplot as plt
RES0.dataframe_by_attrs("R", "E")
plt.matshow(RES0.dataframe_by_attrs("R", "E"), cmap='Blues')


from autograd import grad
gradvv = grad(value_variance_metric, 1)
gradvv(E, R, 0.9)
