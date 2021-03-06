import pandas as pd
import numpy as np

from abc import ABC
from typing import Type, TypeVar# , Union, Mapping
import collections

from functools import reduce
from operator import and_

import utils 
import rv

import warnings
import itertools

import seaborn as sns
greens = sns.light_palette("green", as_cmap=True)

# recipe from https://docs.python.org/2.7/library/itertools.html#recipes
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


def z_mult(joint, masked):
    """ multiply assuming zeros override nans; keep mask."""
    return np.ma.where(joint == 0, 0, joint * masked)
    
def zz1_div(top, bot):
    """ multiply assuming zeros override nans; keep mask."""
    #TODO remove
    # top = np.array(top)
    # bot = np.array(bot)

    rslt = np.ma.divide(top,bot)
    rslt = np.ma.where( np.logical_and(top == 0, bot == 0), 1, rslt)
    return rslt

def D_KL(d1,d2):
    return z_mult(d1, np.ma.log(zz1_div(d1,d2))).sum()


class CDist(ABC): pass
class Dist(CDist): pass

SubCPT = TypeVar('SubCPT' , bound='CPT')


class CPT(CDist, pd.DataFrame, metaclass=utils.CopiedABC):
    PARAMS = {"nfrom", "nto"}
    _internal_names = pd.DataFrame._internal_names + ["nfrom", "nto"]
    _internal_names_set = set(_internal_names)
    
    def __init__(self,*args,**kwargs):
        # print("CPT constructor")
        # self.style.background_gradient(cmap=greens, axis=None)
        pass
        
    # def __call__(self, pmf):
    #     pass
    # def __matmul__(self, other) :
    #     """ Overriding matmul.... """
    #     pass
    
    def flattened_idx(self):
        cols = self.columns.to_flat_index().map(lambda s: s[0])
        rows = self.index.to_flat_index().map(lambda s: s[0])

        return pd.DataFrame(self.to_numpy(), columns = cols, index=rows)
    
    def sample(self, inputs ):
        self.assert_normalized();
        randval = np.random()
        total = 0
        for z,p in self.self[inputs].items():
            if total > randval:
                return z
            else:
                total += p

    @classmethod
    def _from_matrix_inner(cls: Type[SubCPT], nfrom, nto, matrix, multi=True, flatten=False) -> SubCPT:
        def makeidx( vari ):
            if multi and False:
                names=vari.name.split("??")
                
                # maxdepth = utils.tuple_depth(vari.ordered[0])
                # depth = maxdepth - len(names) if flatten else 0
                
                # print("levels", depth)
                # print([ (tuple(utils.flatten_tuples(v, depth)) if type(v) is tuple
                # else (v,) ) for v in vari.ordered ])

                # print(depth)
                print('v', vari.ordered[0])
                print(np.array([ str(v) for v in vari.ordered ]).shape)
                print(names)
                
                # print([ (tuple(utils.flatten_tuples(v, depth)) if type(v) is tuple
                # else (v,) ) for v in [vari.ordered[0]] ])
                # print(np.array([ (tuple(utils.flatten_tuples(v, depth)) if type(v) is tuple
                # else (v,) ) for v in vari.ordered ],).shape)
                # print(names)

                return pd.MultiIndex.from_tuples(
                    [ str(v) for v in vari.ordered ],
                    names=names)
                                    
                # return pd.MultiIndex.from_tuples(
                #     [ (tuple(utils.flatten_tuples(v, depth)) if type(v) is tuple
                #     else (v,) ) for v in vari.ordered ],
                #     names=names)
            else:
                return vari.ordered
        
        return cls(matrix, index=makeidx(nfrom), columns=makeidx(nto), nto=nto,nfrom=nfrom)
        
    @classmethod
    def from_matrix(cls: Type[SubCPT], nfrom, nto, matrix, multi=True, flatten=False) -> SubCPT:
        return cls._from_matrix_inner(nfrom,nto,matrix,multi,flatten).check_normalized()
    
    @classmethod
    def make_stoch(cls: Type[SubCPT], nfrom, nto, matrix, multi=True, flatten=False) -> SubCPT:
        return cls._from_matrix_inner(nfrom,nto,matrix,multi,flatten).renormalize()

    @classmethod
    def from_ddict(cls: Type[SubCPT], nfrom, nto, data) -> SubCPT:
        for a in nfrom:
            row = data[a]
            if not isinstance(row, collections.Mapping):
                try:
                    iter(row)
                except:
                    data[a] = { nto.default_value : row }
                else:
                    data[a] = { b : v for (b,v) in zip(nto,row)}
                    
            total = sum(v for b,v in data[a].items())
            remainder = nto - set(data[a].keys())
            if len(remainder) == 1:
                data[a][next(iter(remainder))] = 1 - total
        
        matrix = pd.DataFrame.from_dict(data , orient='index')
        return cls(matrix, index=nfrom.ordered, columns=nto.ordered, nto=nto,nfrom=nfrom).check_normalized()
        
    @classmethod
    def make_random(cls : Type[SubCPT], vfrom, vto):
        mat = np.random.rand(len(vfrom), len(vto))
        # mat /= mat.sum(axis=1, keepdims=True)
        return cls._from_matrix_inner(vfrom,vto,mat).renormalize()
        
    @classmethod
    def det(cls: Type[SubCPT], vfrom, vto, mapping, **kwargs) -> SubCPT:
        mat = np.zeros((len(vfrom), len(vto)))
        for i, fi in enumerate(vfrom.ordered):
            # for j, tj in enumerate(vto.ordered):
            mapfi = mapping[fi] if isinstance(mapping,dict) else mapping(fi)
            mat[i, vto.ordered.index(mapfi)] = 1
    
        return cls.from_matrix(vfrom,vto,mat, **kwargs)
        # return cls.from_matrix(, index=vfrom.ordered, columns=vto.ordered, nto=vto, nfrom= vfrom)

    @property
    def normalization_error(self):
        return np.where(np.all(np.isfinite(self),axis=1), (np.sum(self, axis=1)-1)**2 ,0).sum()
    
    def assert_normalized(self):
        assert self.normalization_error() < 1E-5, "CPT Unnormalized!"
    
    def check_normalized(self):
        amt = self.normalization_error
        if(amt > 1E-5):        
            warnings.warn("%.2f-Unnormalized CPT"%amt)
            
        return self
        
    def renormalize(self):
        self /= np.sum(self.to_numpy(), axis=1)[:, None]
        return self

## useless helper methods to either use dict values or list.
def _definitely_a_list( somedata ):
    if type(somedata) is dict:
        return list(somedata.values())
    return list(somedata)

# define an event to be a value of a random variale.
class RawJointDist(Dist):
    def __init__(self, data, varlist=None):
        data = np.asarray(data)
        if varlist == None:
            varlist = [ rv.Variable.alph("X%d"%i, n) for i,n in enumerate(data.shape) ]
        self.data = data.reshape(*(len(X) for X in varlist))
        self.varlist = varlist

        
        # if rv.Unit not in varlist:
        #     self.varlist = [rv.Unit] + self.varlist
        #     self.data = self.data.reshape(1, *self.data.shape)
        
        self._query_mode = "dataframe" # query mode can either be
            # dataframe or ndarray

    # Both __mul__ and __rmul__ reqiured to do things like multiply by constants...
    def __mul__(self,other):
        return RawJointDist(self.data * other, self.varlist)
    def __rmul__(self,other):
        return RawJointDist(self.data * other, self.varlist)
        
    def __pos__(self):
        return self.normalize()
    def __floordiv__(self,other):
        return D_KL(self.data,other.data)


    def __repr__(self):
        varstrs = [v.name+"???%d???"%len(v) for v in self.varlist]
        return f"RJD ??[{';'.join(varstrs)}]--{np.prod(self.shape)} params"


    @property
    def shape(self):
        return self.data.shape
        
    def _process_vars(self, vars, given=None):
        if vars is ...:
            vars = self.varlist
            
        if isinstance(vars, rv.Variable) \
            or isinstance(vars, rv.ConditionRequest) or vars is ...:
                vars = [vars]
            
        targetvars = []
        conditionvars = list(given) if given else []

        mode = "join"

        for var in vars:
            if isinstance(var, rv.ConditionRequest):
                if mode == "condition":
                    raise ValueError("Only one bar is allowed to condition")
                    
                mode = "condition"
                targetvars.append(var.target)
                conditionvars.append(var.given)
            else:
                l = (conditionvars if mode == "condition" else targetvars)
                if isinstance(var, rv.Variable):
                    l.append(var)
                    # if mode == "condition":
                    #     conditionvars.append(var)
                    # elif mode == "join":
                    #     targetvars.append(var)
                elif var is ...:
                    l.extend(v for v in self.varlist if v not in l)
                else:
                    raise ValueError("Could not interpret ",var," as a variable")
                
        return targetvars, conditionvars
    
    def _idx(self, var):
        try:
            return self.varlist.index(var)
        except ValueError:
            raise ValueError("The queried varable", var, " is not part of this joint distribution")
        
    def _idxs(self, *varis, multi=False):
        idxs = []
        for V in varis:
            if V in self.varlist and (multi or V not in idxs):
                idxs.append(self.varlist.index(V))
            elif '??' in V.name:
                idxs.extend([v for v in self._idxs(*V.split()) if (multi or v not in idxs)])
            #     for v in V.name.split('??'):
            #         idxs.append([v])
        
        return idxs

    def broadcast(self, cpt : CPT, vfrom=None, vto=None) -> np.array:
        """ returns its argument, but shaped
        so that it broadcasts properly (e.g., for taking expectations) in this
        distribution. For example, if the var list is [A, B, C, D], the cpt
        B -> D would be broadcast to shape [1, 2, 1, 3] if |B| = 2 and |D| =3.
        
        Parameters
        ----
        > cpt: the argument to be broadcast
        > vfrom,vto: the attached variables (supply only if cpt does not have this data)
        """
        if vfrom is None: vfrom = cpt.nfrom
        if vto is None: vto = cpt.nto
        
        # idxf = self.varlist.index(vfrom)
        # idxt = self.varlist.index(vto)
        # 
        # shape = [1] * len(self.varlist)
        # shape[idxf] = len(self.varlist[idxf])
        # shape[idxt] = len(self.varlist[idxt])
        
        # print(vfrom, vto)

        # idxf,idxt = self._idxs(vfrom), self._idxs(vto)
        IDX = self._idxs(vfrom,vto,multi=True)
        UIDX = np.unique(IDX).tolist()
        
        init_shape = [1] * (len(self.varlist)+len(IDX)-len(UIDX))
        
        for j,i in enumerate(IDX):
            init_shape[j] = len(self.varlist[i])

        # print(f,'->', t,'\t',shape)
        # assume cpd is a CPT class..
        # but we don't necessarily want to do this in general
        
        cpt_mat = cpt.to_numpy() if isinstance(cpt, pd.DataFrame) else cpt
        
        # if idxt < idxf:
        #     cpt_mat = cpt_mat.T
                
        cpt_mat = cpt_mat.reshape(*init_shape)
        cpt_mat = np.einsum(cpt_mat, [*IDX,...], [*UIDX, ...])
        cpt_mat = np.moveaxis(cpt_mat, np.arange(len(UIDX)), UIDX)

        return cpt_mat
        # return cpt_mat.reshape(*end_shape)

    def normalize(self):
        self.data /= self.data.sum()
        return self
    
    def conditional_marginal(self, vars, query_mode=None):
        if query_mode is None: query_mode = self._query_mode
        # if coordinate_mode is "joint": query_mode = "ndarray"
        
        # print(type(vars), vars, isinstance(vars, rv.Variable))
        targetvars, conditionvars = self._process_vars(vars)

        idxt = self._idxs(*targetvars)
        idxc = self._idxs(*conditionvars)
        IDX = idxt + idxc
        
        # sum across anything not in the index
        joint = self.data.sum(axis=tuple(i for i in range(len(self.varlist)) if i not in IDX))
        
        # duplicate dimensions that occur multiple times by 
        # an einsum diagonalization...
        joint_expanded = np.zeros([self.data.shape[i] for i in IDX])
        np.einsum(joint_expanded, IDX, np.unique(IDX).tolist())[...] = joint
        
        if len(idxc) > 0:
            # if idxt is first...
            normalizer = joint_expanded.sum(axis=tuple(i for i in range(len(idxt))), keepdims=True)
            
            #if idxt is last...
            # normalizer = joint_expanded.sum(axis=tuple(-i-1 for i in range(len(idxt))), keepdims=True)
        
            # return joint_expanded / normalizer
            matrix = joint_expanded / normalizer;
            if query_mode == "ndarray":
                return matrix
            elif query_mode == "dataframe":
                vfrom = reduce(and_,conditionvars)
                vto = reduce(and_,targetvars)
                mat2 = matrix.reshape(len(vto),len(vfrom)).T

                return CPT.from_matrix(vfrom,vto, mat2,multi=False)
        else:
            # return joint_expanded
            if query_mode == "ndarray":
                return joint_expanded
            elif query_mode == "dataframe":
                mat1 = joint_expanded.reshape(-1,1).T;
                return CPT.from_matrix(rv.Unit, reduce(and_,targetvars), mat1,multi=False)
                
    # returns the marginal on a variable
    def __getitem__(self, vars):
        return self.conditional_marginal(vars, self._query_mode)
    
    
    def prob_matrix(self, *vars, given=None):
        """ A global, less user-friendly version of 
        conditional_marginal(), which keeps indices for broadcasting. """        
        tarvars, cndvars = self._process_vars(vars, given=given)
        idxt = self._idxs(*tarvars)
        idxc = self._idxs(*cndvars)
        IDX = idxt + idxc
        
        N = len(self.varlist)
        dim_nocond = tuple(i for i in range(N) if i not in idxc )
        dim_neither = tuple(i for i in range(N) if i not in IDX ) # sum across anything not in the index
        # print("dim_nocond", dim_nocond, "dim_neither", dim_neither, "shape", self.data.shape)
        collapsed = self.data.sum(axis=dim_neither, keepdims=True)
        
        if len(cndvars) > 0:
            # collapsed /= collapsed.sum(axis=dim_nocond, keepdims=True)
            collapsed = np.ma.divide(collapsed, collapsed.sum(axis=dim_nocond, keepdims=True))
            
        return collapsed

            
    def H(self, *vars, base=2, given=None):
        """ Computes the entropy, or conditional
        entropy of the list of variables, given all those
        that occur after a ConditionRequest. """
        
        return - (np.ma.log( self.prob_matrix(*vars, given=given) ) * self.data).sum() / np.log(base)
        
        ## The expanded version looks like this, but is 
        ## a bit slower and not really simpler.
        # collapsed = self.prob_matrix(vars)
        # surprise = - np.ma.log( collapsed ) / np.log(base)
        # E_surprise = surprise.filled(0) * self.data
        # return E_surprise.sum()
    
    def I(self, *vars, given=None):
        tarvars, cndvars = self._process_vars(vars, given)
        
        sum = 0
        # n = len(tarvars)
        
        for s in powerset(tarvars):
            # print(s, (-1)**(n-len(s)), self.H(*s, given=cndvars))
            sum += (-1)**(len(s)+1) * self.H(*s, given=cndvars) # sum += (-1)**(n-len(s)+1) * self.H(*s, given=cndvars)
        return sum
    
    # def _info_in(self, vars_in, vars_fixed):
        # return self.H(vars_in | vars_fixed)
    # 
    def iprofile(self) :
        """
        Returns a tensor of shape 2*2*2*...*2, one dimension for each
        variable. For example, 
            00000 is going to always have zero.
            01000 is the information H(X1 | X0, X2, ... Xn)
            11000 is the conditional mutual information I(X1; X2 | ...)
            
        """
        for S in powerset(self.varlist):
            pass
    
    
    def info_diagram(self, X, Y, Z=None):
        # import matplotlib.pyplot as plt
        from matplotlib_venn import venn3
         
         
        # H = self.H
        I = self.I
        
        infos = [I(X|Y,Z), I(Y|X,Z), I(X,Y|Z), I(Z|X,Y), I(X,Z|Y), I(Y,Z|X), I(X,Y,Z) ]
        infos = [round(i, 3) for i in infos]
        # infos = [int(round(i * 100)) for i in infos]
        # Make the diagram
        v = venn3(subsets = infos, set_labels=[X.name,Y.name,Z.name]) 
        return v

    #################### CONSTRUCTION ######################
                
    @staticmethod
    def unif( vars) -> 'RawJointDist':
        varlist = _definitely_a_list(vars)
        data = np.ones( tuple(len(X) for X in varlist) )
        return RawJointDist(data / data.size, varlist)

    @staticmethod
    def random( vars) -> 'RawJointDist':
        varlist = _definitely_a_list(vars)
        data = np.random.rand( *[len(X) for X in varlist] )
        return RawJointDist(data / np.sum(data), varlist)


# 
# class CoreJointDist(RawJointDist):
#     def __init__(self, data, varlist):
# 
#         self.redvars = [v for v in varlist if '??' in v.name]
#         self.ghostvars = [v for v in varlist if '??' in v.name]
# 
#         self.rvlookup = {v.name: v for v in self.redvars}
# 
#         missing = [n for v in self.ghostvars for n in v.name.split('??') if n not in self.redvars]
#         assert len(missing)==0, "Missing Components: "+repr(missing)
# 
#         super().__init__(self, data, self.redvars)
# 
#     def vsplit(self, *vars):
#         for V in vars:
#             for v in V.name.split('??'):
#                 yield self.rvlookup[v]
# 
#     def _process_vars(self, vars, given=None):
#         if vars is ...:
#             vars = self.redvars
# 
#         if isinstance(vars, rv.Variable) \
#             or isinstance(vars, rv.ConditionRequest):# or vars is ...:
#                 vars = [vars]
# 
#         targetvars = []
#         conditionvars = list(given) if given else []
# 
#         mode = "join"
# 
#         for var in vars:
#             if isinstance(var, rv.ConditionRequest):
#                 if mode == "condition":
#                     raise ValueError("Only one bar is allowed to condition")
# 
#                 mode = "condition"
#                 targetvars.append(var.target)
#                 conditionvars.append(var.given)
#             else:
#                 l = (conditionvars if mode == "condition" else targetvars)
#                 if isinstance(var, rv.Variable):
#                     l.append(*self.vsplit(var))
#                 elif var is ...:
#                     l.extend(v for v in self.varlist if v not in l)
#                 else:
#                     raise ValueError("Could not interpret ",var," as a variable")
# 
#         return targetvars, conditionvars
# 
    # def _idxs(self, var):
    #     try:
    #         return self.varlist.index(var)
    #     except ValueError:
    #         raise ValueError("The queried varable", var, " is not part of this joint distribution")

    # def broadcast(self, cpt, vfrom=None, vto=None) -> np.array:
    #     """ returns its argument, but shaped
    #     so that it broadcasts properly (e.g., for taking expectations) in this
    #     distribution. For example, if the var list is [A, B, C, D], the cpt
    #     B -> D would be broadcast to shape [1, 2, 1, 3] if |B| = 2 and |D| =3.
    # 
    #     Parameters
    #     ----
    #     > cpt: the argument to be broadcast
    #     > vfrom,vto: the attached variables (supply only if cpt does not have this data)
    #     """
    #     if vfrom is None: vfrom = cpt.nfrom
    #     if vto is None: vto = cpt.nto
    # 
    #     idxf = self.varlist.index(vfrom)
    #     idxt = self.varlist.index(vto)
    # 
    #     shape = [1] * len(self.varlist)
    #     shape[idxf] = len(self.varlist[idxf])
    #     shape[idxt] = len(self.varlist[idxt])
    # 
    #     # print(f,'->', t,'\t',shape)
    #     # assume cpd is a CPT class..
    #     # but we don't necessarily want to do this in general
    # 
    #     cpt_mat = cpt.to_numpy() if isinstance(cpt, pd.DataFrame) else cpt
    #     if idxt < idxf:
    #         cpt_mat = cpt_mat.T
    # 
    #     return cpt_mat.reshape(*shape)
