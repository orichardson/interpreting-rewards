import numpy as np
from scipy.special import logsumexp
# import pandas as pd
# import gym


# ρ = pd.DataFrame([[0,1], [2,1]],
#     columns=["a1", "a2"],  index=["s1", "s2"])


from rv import Variable as Var
from dist import CPT

kd = dict(keepdims=True)

class Env:
    def __init__(self, States, Acts, Transitions):
        self.S = States
        self.A = Acts
        self.T = Transitions # (A, S) -> ΔS
        
    def __getattr__(self, key):
        if key.endswith("shape"):
            firstpart = key[:-5]
            if firstpart in ["T", "T_", "τ", "transition_", "SAS", "SAS_"]: 
                return (self.nS, self.nA, self.nS)
            elif firstpart in ["Q", "pi", "pi_", "π", "policy_", "SA", "SA_","SA1"]:
                return (self.nS, self.nA, 1)
            elif firstpart in ["V", "S", "π", "D_"]:
                return (self.nS, 1, 1)
            
    def random_policy(self, det=False):
        raw = np.random.exponential(size=self.pi_shape)
        if det:
            ###___ PROBLEM with the below: gives all maxima, not just first.___
            # return 0.0 + (raw == raw.max(axis=1, **kd))
            ###__ SO INSTEAD, do this.
            res = np.zeros(raw.shape)
            for (s,_),v in np.ndenumerate(raw.argmax(axis=1)):
                res[s,v,0] = 1
            return res
                
        else:
            return raw / raw.sum(axis=1, **kd)
    
    @property
    def novelty(self):
        # return env.TT
        # (s,a,s') =>  - log T(s' | a,s)
        TT = self.TT
        with np.errstate(divide="ignore"):
            return np.where(TT==0, 0, np.log(1/self.TT)*self.TT)
        
    

    
    @property
    def nS(self): return len(self.S)
    @property
    def nA(self): return len(self.A)

    @property
    def TT(self):
        return self.T.to_numpy().reshape(self.Tshape)


    @staticmethod
    def generate(A,S, spec="random"):

        vS = S if isinstance(S, Var) else Var(S, name="S")
        vA = A if isinstance(A, Var) else Var(A, name="A")
        vS2 = vS.copy(name=vS.name+"'")
        
        cptshape = (len(S)*len(A), len(S))
        if isinstance(spec, str):
            if spec.startswith("random"): tensor = np.random.exponential(size=cptshape)
            elif spec.startswith("unif"): tensor = np.ones(cptshape)
            
            if "det" in spec:
                tensor = (tensor == tensor.max(axis=-1, **kd))+0.0
                
        else:
            try:
                tensor = spec(*cptshape)
            except TypeError:
                raise ValueError("spec `%s` neither string nor callable" % repr(spec))
        
        return Env(vS, vA, CPT.make_stoch(vS & vA, vS2, tensor ) )
        
    
    @staticmethod
    def make_contextual_bandit(A, S, method="random"):
        return Env.generate(A,S,"unif")
    
    @staticmethod
    def make_bandit(A):
        return Env.make_contextual_bandit(A, {"s"})
     
    @staticmethod
    def make_from_T_tensor(T):
        """ T.shape = (S, A, S') """
        nS, nA, nS2 = T.shape();
        assert nS == nS2, "Transitions must source (%d) and target (%d) state space of same size" % (nS, nS2)
        
        vS = Var.alph("S", nS)
        vS2 = vS.copy(name= "S'")
        vA = Var.alph("A", nA)
        return Env(vS, vA, CPT.make_stoch(vS & vA, vS2, T.reshape(nS*nA, nS2)))

class GridWorld(Env):
    
    # def __init__():
    #     pass

    def draw(self, **kwargs):
        
        from matplotlib import cm, pyplot as plt
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        
        fig, ax = plt.subplots(figsize=(self.W,self.H))
        ax.axis('off')
        # N = self.W * self.H
        
        for descr,array in kwargs.items():
            # if descr is None and np.prod(array.shape[1:]) == 1: descr = "states"
            if descr == "descr":
                print(array)
                ax.set_title(array)
                
            if "states" in descr: # just a state array
                M = array.reshape(self.W, self.H).T
                norm = cm.colors.Normalize(vmax=abs(M).max()+0.001, vmin=-abs(M).max()-0.001)
                
                ###_ SET UP COLORMAP _###
                colors = np.vstack([
                        cm.get_cmap("Reds_r",128)(np.linspace(0,1,128)),
                        cm.get_cmap("Greens",128)(np.linspace(0,1,128))
                    ])
                colors[:,3] = np.hstack((np.linspace(1,0,128), np.linspace(0,1,128)))                
                red_green_cmap = ListedColormap(colors, name="RedGreen")
                ###__###############__###
                
                ax.matshow(M,cmap=red_green_cmap, norm=norm)
                
            if "policy" in descr:
                pi = array.reshape(-1,4)

                X,Y = np.meshgrid(range(self.W), range(self.H))
                X,Y = X.T, Y.T

                dx = np.array([0,-1,0,1]).reshape(1, 4)
                dy = np.array([-1,0,1,0]).reshape(1, 4)
                
                U = (pi * dx).sum(axis=1)
                V = -(pi * dy).sum(axis=1)
                with np.errstate(divide='ignore'):
                    entropy = (- pi * np.where(pi==0, 0, np.log(pi))).sum(axis=1)
                    
                    
                ###_ SET UP COLORMAP _###
                colors = cm.get_cmap("Blues",256)(np.linspace(0,1,256))
                colors[:,3] = np.linspace(0,1,256)                
                arrow_cmap = ListedColormap(colors, name="arrow_cmap")
                ###__###############__###

                for a in range(4):
                    field = pi[:,a,...].reshape(self.W, self.H)
                    ax.quiver(X + dx[0, a]/4,Y +dy[0,a]/4, field*dx[0,a], -field*dy[0,a], field, 
                              cmap=arrow_cmap,units='x', pivot='mid', headlength=3,headaxislength=2, scale=2.5)
                    
                    
                ax.quiver(X,Y,U,V,
                              #-entropy, cmap='binary',
                              pivot='mid', units='x', 
                              scale=1.1,headlength=5,headwidth=4,width=0.09,headaxislength=3.5,
                              edgecolor=(1,1,1,0.5), linewidth=0.5, alpha=0.5)

        
        # else:
        #     raise NotImplementedError()
    
    @staticmethod
    def make(width, height, noise=0) -> 'GridWorld':
        S = Var.alph("X", width) & Var.alph("Y", height)
        A = Var(["up","left", "down", "right"], name="A")
        
        dx = [0,-1,0,1]
        dy = [-1,0,1,0]
        # δ = [-width, +width, +1, -1]
        # dim = [0,0,1,1]
        
        def tgt(x,y,aidx):
            newx = np.clip(x + dx[aidx], 0, width-1)
            newy = np.clip(y + dy[aidx], 0, height-1)
            return newy + newx * height # this is weird, but due to ordering
                
        T = np.zeros((len(S), len(A), len(S)))
        for sidx,(x,y) in enumerate(S.ordered):
            x,y = [int(c[1:]) for c in (x,y)]
            for a in range(len(A)):
                T[sidx, a, tgt(x,y,a)] = 1 - noise
                T[sidx, a, tgt(x,y,(a+1)%4)] += noise/2
                T[sidx, a, tgt(x,y,(a-1)%4)] += noise/2
              
        GW = GridWorld(S, A, CPT.from_matrix(S&A,S.copy(name= "S'"),
                                                T.reshape( len(S)*len(A), len(S)) ))
        GW.W = width
        GW.H = height
        return GW
    
    
A3 = Var.alph("A",3)
S2 = Var.alph("S",2)
Eband = Env.make_bandit(A3)
Ectxb = Env.make_contextual_bandit(A3, S2)
Erand = Env.generate(A3, S2, spec="random")
Ebigd = Env.generate(Var.alph("A", 3), Var.alph("S", 20), spec="random.det")

# def optimal_policy( r, ENV ):
# def _preprocess(env, R, pi)
    
def value_iter(env, R, γ, iters=100, init=None, temperature=0): 
    V = np.zeros(env.Vshape) if init is None else init
    for it in range(iters):
        # V = max_a \Ex_{s'~p|a,s} [ r(a,s) + \gamma V(s') ]
        # Qs =  (env.TT * (R + γ * V.T)).sum(axis=2,**kd)
        V = t_max( Q(env, R, V, γ), temperature, axis=1)

    return V *  (1 - γ) / (1 - np.power(γ, iters))

def value_iter_recencybias(env, R, γ, iters=100, init=None, temperature=0): 
    V = np.zeros(env.Vshape) if init is None else init
    for it in range(iters):
        # V = (env.TT * (R * (1-γ) + γ * V.T)).sum(axis=2,**kd).max(axis=1,**kd)
        Qs = (env.TT * (R * (1-γ) + γ * V.T)).sum(axis=2,**kd)
        V = t_max( Qs, temperature, axis=1)

    return V

def policy_eval(env:Env, pi, R, γ, iters=100, init=None):
    # pi : (S, A)
    V = np.zeros(env.Vshape) if init is None else init
    pi_arr = np.array(pi).reshape(env.πshape)
    
    for it in range(iters):
        # _____ RECENCY BIAS _____
        # V = (pi_arr * env.TT * (R + γ * V.T)).sum(axis=(1,2)) / (1 + γ)
        #!! WORRY: (1+gamma) is not the usual (1-\gamma)...
        
        # THE USUAL VERSION
        V = (pi_arr * env.TT * (R + γ * V.T)).sum(axis=(1,2)) 
        
    # return V
    return V *  (1 - γ) / (1 - np.power(γ, iters))
    
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
    V = np.zeros(env.Vshape) if init is None else init
    pi_arr = np.array(pi).reshape(env.πshape)
    
    for it in range(iters):
        V = (pi_arr * env.TT * ((1 - γ) * R + γ * V.T)).sum(axis=(1,2))        
        
    return V


def Q(env, R, V, γ):
    return (env.TT*(R + γ * V)).sum(axis=2,**kd) 
    # / (1 + γ)
    
def best_policy( env, R, V, γ, temperature=0):
    return t_argmax(Q(env, R, V, γ), temperature, axis=1)
    
    # Qmaxima = (Qvals == Qvals.max(axis=1,**kd))+0.0 if temperature == 0  \
    #     else np.exp( Qvals / temperature )        
    # return Qmaxima / (Qmaxima.sum(axis=1,**kd))

def event_joint_iter(env, init=None, iters=100):
    """
    One issue:     
    """
    dist = np.ones(env.SAshape) / np.prod(env.SAshape) if init is None \
        else np.array(init).reshape(env.SAshape)
    
    # iterative fixed pt computation (i.e., right eigenvector)
    for it in range(iters):
        dist = (env.TT * dist).sum(axis=0,**kd).transpose()
    
    return dist


def visitation_iter(env, pi, init=None, iters=100):
    dist = np.ones(env.Sshape) / np.prod(env.Sshape) if init is None \
        else np.array(init).reshape(env.Sshape)
    pi = np.array(pi).reshape(env.pi_shape)
    
    # iterative fixed pt computation (i.e., right eigenvector)
    for it in range(iters):
        dist = (env.TT * dist * pi).sum(axis=(0,1), **kd).transpose()
    
    return dist


def fwd(env, R, γ, policy_improve_iters=0, val_iters=100, temp=0):
    V = value_iter(env, R, γ, val_iters, temperature=temp)
    π = best_policy(env, R, V, γ, temperature=temp)
    for i in range(policy_improve_iters):
        V = policy_eval(env, π, R, γ, iters=val_iters, init=V)
        π = best_policy(env, R, V, γ, temperature=temp)
    return π

def t_max(arr, temp = 0, axis=None):
    if temp == 0:
        return np.asarray(arr).max(axis=axis, **kd)
    else:
        ### Would like to do this, but numerically unstable. 
        # return temp * np.log( np.exp(arr/temp).sum(**ufunc_kwargs) )
        ### Unfortunately scipy's logsumexp deals with temperature wrong.
        return temp*logsumexp(np.divide(arr, temp), axis=axis, keepdims=True)
        
    
def t_argmax(arr, temp=0, axis=None):
    if temp == 0:
        if not isinstance(arr,np.ndarray):
            arr = np.asarray(arr)
        almost = (arr == arr.max(axis=axis, **kd)) +0.0
        return almost / almost.sum(axis=axis, **kd)

    else:
        ascaled = np.divide(arr, temp)
        almost = np.exp(ascaled - logsumexp(ascaled, axis=axis, **kd))
        return almost / almost.sum(axis=axis, **kd)
    



    # (s,a) =>  E_{s'} - log T(s' | a,s)
    # return (-np.log(env.TT)*env.TT).sum(axis=2)
    # raise NotImplementedError()
    
def broadcompress(array, tol = 1E-8):
    if not isinstance(array,np.ndarray):
        array = np.asarray(array)
    for d in array.shape:
        array = np.moveaxis(array, 0, -1)
        if np.all(np.var(array,axis=0) <= tol):
            array = array[(0,),...]
    return array
            

def canonicalize(R, DA=None, DS=None):
    # R +  Ex_[ gamma R(s', A, S') - R(s, A, S') + gamma R(S,A,S') ]
    R = np.asarray(R)
    if DA is None: DA = np.ones()
    DA = np.asarray(DA) / np.sum(DA, axis=(0,1))
    if DS is None: DS = np.ones()
    DS = np.asarray(DS) / np.sum(DS, axis=(0,1))
    
    raise NotImplementedError()
    
def logZ(R, temp):
    return temp * np.log(np.exp(R / temp).sum(axis=1))
    
def policyReward(pi,  temp=0):
    with np.errstate(divide='ignore'):
        infocost = np.log(pi)
        
    am = t_argmax(infocost, temp, axis=1)
    val = t_max(pi, temp, axis=1)
    # val = am.max(axis=1, **kd)
    return am - val
    
    
def MCE_IRL(env:Env, pi, γ, lr=1, lr_decay=0.9, 
            policy_improve_iters=0,
            value_iters = 100,
            visit_iters = 100,
            temperature = 0,
            iters=100):
    pi = np.array(pi).reshape(env.SA_shape)
    R = np.zeros(env.Tshape)
    D_want = visitation_iter(env, pi, iters=visit_iters)

    
    for it in range(iters):
        pi_R = fwd(env, R, γ, policy_improve_iters, value_iters, temp=temperature)
        D_R = visitation_iter(env, pi_R, iters=visit_iters)
        R += lr * ( D_want * pi - D_R * pi_R)
        lr *= lr_decay
        
    return R