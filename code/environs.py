
from rv import Variable as Var
from dist import CPT
# import numpy as np
import torch
import numpy

from utils import matches_any, nparray_of

            
kd = dict(keepdims=True)

class Env:
    def __init__(self, States, Acts, Transitions):
        self.S = States
        self.A = Acts
        self.T = Transitions # (A, S) -> ΔS
        self._T = torch.tensor(Transitions.to_numpy()).reshape(self.Tshape) # tensor
        
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
        raw = numpy.random.exponential(size=self.pi_shape)
        if det:
            ###___ PROBLEM with the below: gives all maxima, not just first.___
            # return 0.0 + (raw == raw.max(axis=1, **kd))
            ###__ SO INSTEAD, do this.
            res = numpy.zeros(raw.shape)
            for (s,_),v in numpy.ndenumerate(raw.argmax(axis=1)):
                res[s,v,0] = 1
            return torch.tensor(res)
                
        else:
            return torch.tensor(raw / raw.sum(axis=1, **kd))
    
    @property
    def novelty(self):
        # return env.TT
        # (s,a,s') =>  - log T(s' | a,s)
        TT = self.TT
        with numpy.errstate(divide="ignore"):
            return torch.where(TT==0, 0., -torch.log( torch.where(TT==0, 1., TT))*TT)
        
    

    
    @property
    def nS(self): return len(self.S)
    @property
    def nA(self): return len(self.A)

    @property
    def TT(self):
        # for now, just validate nothing is wrong
        assert numpy.allclose(self._T, 
            self.T.to_numpy().reshape(self.Tshape)), "Transition Dynamics changed..."
        
                    
        return self._T
        

    @staticmethod
    def generate(A,S, spec="random"):

        vS = S if isinstance(S, Var) else Var(S, name="S")
        vA = A if isinstance(A, Var) else Var(A, name="A")
        vS2 = vS.copy(name=vS.name+"'")
        
        cptshape = (len(S)*len(A), len(S))
        if isinstance(spec, str):
            if spec.startswith("random"): array = numpy.random.exponential(size=cptshape)
            elif spec.startswith("unif"): array = numpy.ones(cptshape)
            
            if "det" in spec:
                array = (array == array.max(axis=-1, **kd))+0.0
                
        else:
            try:
                array = spec(*cptshape)
            except TypeError:
                raise ValueError("spec `%s` neither string nor callable" % repr(spec))
        
        return Env(vS, vA, CPT.make_stoch(vS & vA, vS2, array ) )
        
    
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




# _trioff = {
#     'up' : [ numpy.array([0,0]), numpy.array([1,0]) ],
#     'down' : [ numpy.array([0,1]), numpy.array([1,1]) ],
#     'left' : [ numpy.array([0,0]), numpy.array([0,1]) ],
#     'right' : [ numpy.array([1,0]), numpy.array([1,1]) ]
# }
# def _tricalc(i,j, k):
#     mid = numpy.array([i,j]) + 0.5
#     return [mid] + 
# 

class GridWorld(Env):
    
    # def __init__():
    #     pass

    def draw(self, ax=None, **kwargs):
        
        from matplotlib import cm, pyplot as plt
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        
        ###_ SET UP COLORMAP _###
        colors = numpy.vstack([
                cm.get_cmap("Reds_r",128)(numpy.linspace(0,1,128)),
                cm.get_cmap("Greens",128)(numpy.linspace(0,1,128))
            ])
        colors[:,3] = numpy.hstack((numpy.linspace(1,0,128), numpy.linspace(0,1,128)))                
        red_green_cmap = ListedColormap(colors, name="RedGreen")
        ###__###############__###
        ###_ SET UP COLORMAP _###
        colors = numpy.vstack([
                cm.get_cmap("Oranges_r",128)(numpy.linspace(0,1,128)),
                cm.get_cmap("Blues",128)(numpy.linspace(0,1,128))
            ])
        colors[:,3] = numpy.hstack((numpy.linspace(1,0,128), numpy.linspace(0,1,128)))
        orange_blue_cmap = ListedColormap(colors, name="OrangeBlue")
        ###__###############__###
        ###_ SET UP COLORMAP _###
        colors = cm.get_cmap("Blues",256)(numpy.linspace(0,1,256))
        colors[:,3] = numpy.linspace(0,1,256)                
        arrow_cmap = ListedColormap(colors, name="arrow_cmap")
        ###__###############__###
        
        if ax == None:
            fig, ax = plt.subplots(figsize=(self.W,self.H))
            # fig, ax = plt.subplots()
        ax.axis('off')
        
        
        ax.matshow(numpy.zeros((self.W, self.H)).T, alpha=0)
        # N = self.W * self.H
        
        for descr,array in kwargs.items():
            # if descr is None and np.prod(array.shape[1:]) == 1: descr = "states"
            if descr == "descr":
                print(array)
                ax.set_title(array)
                
            # just a state array
            if matches_any(descr, "states", "grid"):                
                M = nparray_of(array).reshape(self.W, self.H).T
                norm = cm.colors.Normalize(vmax=abs(M).max()+0.001, vmin=-abs(M).max()-0.001)
                
                ax.matshow(M,cmap=red_green_cmap, norm=norm)
                
            #draw triangulation
            if matches_any(descr, "trigrid", "tris", "savals"):
                M = nparray_of(array).reshape(self.W, self.H, self.nA)
                norm = cm.colors.Normalize(vmax=abs(M).max()+0.001, vmin=-abs(M).max()-0.001)
                
                
                X1,Y1 = numpy.meshgrid(range(self.W+1), range(self.H+1))
                Xm, Ym = numpy.meshgrid(range(self.W), range(self.H))
                X = numpy.concatenate((X1.T.flatten()-0.5, Xm.T.flatten()))
                Y = numpy.concatenate((Y1.T.flatten()-0.5, Ym.T.flatten()))
                N = numpy.size(X1)
                
                triangles = []
                COL = self.H+1
                for (x,y) in zip(Xm.T.flatten(), Ym.T.flatten()): 
                    i = numpy.ravel_multi_index((x,y), (self.W,self.H))
                    ul = y + COL*x
                    triangles.append( [N+i,  ul+COL, ul] ) # top
                    triangles.append( [N+i,  ul, ul+1] ) # left
                    triangles.append( [N+i,  ul+1, ul+COL+1] ) # bottom
                    triangles.append( [N+i,  ul+COL+1, ul+COL] ) # right
                


                ax.tripcolor(X,Y,triangles, facecolors=M.flatten(),
                    # cmap='Blues', 
                    cmap=orange_blue_cmap,
                    norm=norm,
                    # edgecolors='k', 
                    alpha=0.8)
                
            # draw quiver
            if matches_any(descr, "policy", "π", "flow"):
                pi = nparray_of(array).reshape(-1,4)

                X,Y = numpy.meshgrid(range(self.W), range(self.H))
                X,Y = X.T, Y.T

                dx = numpy.array([0,-1,0,1]).reshape(1, 4)
                dy = numpy.array([-1,0,1,0]).reshape(1, 4)
                
                U = (pi * dx).sum(axis=1)
                V = -(pi * dy).sum(axis=1)
                with numpy.errstate(divide='ignore'):
                    # entropy = (- pi * numpy.where(pi==0, 0, numpy.log(pi))).sum(axis=1)
                    pass

                for a in range(4):
                    field = pi[:,a,...].reshape(self.W, self.H)
                    ax.quiver(X + dx[0, a]/4,Y +dy[0,a]/4, field*dx[0,a], -field*dy[0,a], field, 
                              cmap=arrow_cmap,units='x', pivot='mid', headlength=3,headaxislength=2, scale=2.5)
                    
                    
                ax.quiver(X,Y,U,V,
                              #-entropy, cmap='binary',
                              pivot='mid', units='x', 
                              scale=1.1,headlength=5,headwidth=4,width=0.09,headaxislength=3.5,
                              edgecolor=(1,1,1,0.5), linewidth=0.5, alpha=0.5, zorder=4)

        
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
            newx = numpy.clip(x + dx[aidx], 0, width-1)
            newy = numpy.clip(y + dy[aidx], 0, height-1)
            return newy + newx * height # this is weird, but due to ordering
                
        T = numpy.zeros((len(S), len(A), len(S)))
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
