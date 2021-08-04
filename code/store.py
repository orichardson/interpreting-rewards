import numpy as np
from dist import RawJointDist as RJD
from environs import Env
# from collections import frozenset as fz

class TensorLibrary:
    def __init__(self, shape=(-1,), decoder=None):
        # self.tensordata = dists
        self.ushape = shape
        self.decoder = decoder
        self.tensordata = {} # frozenset( str | (k:v) )  =>  ℝ(ushape)    
        # self.ushape = M.genΔ(repr=store_repr).data.reshape(shape).shape

    def __getattr__(self, name):
        return getattr(LView(self), name)
        # if name[0] == '_':
            # pass
        # return View(self).__getattr__(name)
        
    # def __iadd__(self, other):
        # pass
        
    def items(self):
        return self.tensordata.items()
        
    def keys(self):
        return self.tensordata.keys()
        

    def __call__(self, *posspec, **kwspec):
        return LView(self)(*posspec, **kwspec)


    def _validate(self, value):
        if self.shape:
            try:
                argvalue = value
                if isinstance(value, RJD): value = value.data
                elif isinstance(value, Env): value = value.TT
                
                value = np.asarray(value) #try this
                return value.reshape(self.ushape)
            except:
                raise TypeError("Only tensors of shape "+str(self.ushape)+" allowed; \"", argvalue, "\" could not be reshaped this way")
        else:
            return value

    def _decode(self, stored_tensor):
        if self.decoder:
            return self.decoder(stored_tensor)
        else:
            return stored_tensor

    def __setitem__(self, key, val):
        self.tensordata[frozenset(key)] = self._validate(val)
        
    def copy(self):
        tl = TensorLibrary(self.shape, self.decoder)
        tl.tensordata = dict(self.tensordata)
        return tl

    #TODO: niciefy this
    # def __repr__(self):
    #     keystr = '; '.join(
    #         (str(s[0])+"="+str(s[1]) if isinstance(s,tuple) and len(s)==2 \
    #         else s) for s in self.tensordata.keys())
    #     return "<DistLib with keys {%s}>"%s
    def __iter__(self):
        return iter(LView(self))

    def __pos__(self):
        return +LView(self)
        
    def __len__(self):
        return len(self.tensordata)
        
    # def __repr__(self):
    #     return 

def _has2(v):
    return isinstance(v, tuple) and len(v) == 2
def _mixed2dict( mixed_selector, default ):
    return dict((x if _has2(x) else (x, default)) for x in mixed_selector)

def valid_selector(s):
    if _has2(s):
        return valid_selector(s[0])
    return not (isinstance(s,str) and s[0] == '_')


from itertools import chain
from inspect import getsource
class LView:
    def __init__(self, library, *selector, **kwselect):
        self._lib = library
        self._most_recent_tag = selector[-1] if len(selector) > 0 else None
        self._filters = kwselect.get('_filters', [])
        self._sel  = frozenset(chain(selector,
                        filter(valid_selector, kwselect.items())))
        self._cached = list(self._consist_from_lib())


    def _consist_from_lib(self):
        for k,d in self._lib.tensordata.items():
            # if self._sel.issubset(k) \
            if all(((t in k or t[0] in k) if _has2(t) else (t in k)) for t in self._sel ) \
                    and all(f(k) for f in self._filters):
                yield k,d

    def along(self, axis, return_tags=False):
        """
        Allows you to simultaneously filter by an attribute and sort by it, 
        optionally in decending order.
        
        E.g., `store.along('-x')`
        # returns an iterator of tensors with the x attribute, sorted in reverse
        """
        reverse=False
        if isinstance(axis,str) and axis[0] in '+-':
            reverse = (axis[0] == '-')
            axis = axis[1:]

        values = []
        for S,d in (self._cached if self._cached else self._consist_from_lib()):
             v = next((atom[1] for atom in S if atom[0] == axis), None)
             if v is not None:
                 values.append( (v,(S,self._lib._decode(d))) )

        return (((S,d) if return_tags else d) for v,(S,d) in sorted(values, reverse=reverse))

    def filter(self, f):
        return LView(self._lib, *self._sel, _filters=[*self._filters, f])
        
    @property
    def tags(self):
        return set( k for S in self.matches for k in _mixed2dict(S, None).keys() )

    @property
    def tensors(self):
        for s,d in self:
            yield d

    @property
    def matches(self):
        for s,d in self.raw:
            yield s

    def without(self, tag, **kwargs):
        return self.filter(lambda taglist: tag not in _mixed2dict(taglist, ...))

    def set(self, dist):
        self._lib.tensordata[self._sel] = self._lib._validate(dist)
        
    def tagAll(self, *tags, **kwtags):
        newdict = {}
        for S,t in self.raw:
            # make sure new values are overriden
            S_preempt_duplicates = [ k for k in S if not(_has2(k) and valid_selector(k) and  k[0] in kwtags) ]
            
            newS = frozenset(chain(S_preempt_duplicates,tags,\
                filter(valid_selector, kwtags.items())))
            # newS = S.union(tags, filter(valid_selector, kwtags.items()))
            newdict[newS] = self._lib.tensordata[S] 
            del self._lib.tensordata[S]
        
        # print(newdict)
        self._lib.tensordata.update(newdict)
        self._sel = self._sel.union(tags, filter(valid_selector, kwtags.items()))

    # def __iadd__(self, datum):
    #     self.set(datum)

    @property
    def raw(self):
        for s,d in (self._cached if self._cached else self._consist_from_lib()):
            yield s,d
    
    def __iter__(self):
        for s,d in (self._cached if self._cached else self._consist_from_lib()):
            yield s, self._lib._decode(d)

    def __pos__(self):
        lubs = []
        for s,d in self.raw:
            if all(s.issubset(l) for l in lubs):
                lubs = [ s ]
            elif not any(l.issubset(s) for l in lubs):
                lubs.append(s)
        if len(lubs) != 1:
            raise ValueError("No minimal distribution in this view! (there are %d)"%len(lubs))

        return self._lib._decode(self._lib.tensordata[lubs[0]])
        # return self._lib.tensordata[self._sel]
        # return next(iter(self.μs))

    def __repr__(self):
        selectorstr = '; '.join(
            (str(s[0])+"="+str(s[1]) if isinstance(s,tuple) and len(s)==2 \
            else str(s)) for s in self._sel)

        if self._filters:
            selectorstr += " | <%d filters>" % len(self._filters)

        return "LibView { %s } (%d matches)" % (selectorstr, len(self._cached))


    def __getattr__(self, name):
        if frozenset([*self._sel, name]) in self._lib.tensordata:
            return self._lib._decode(self._lib.tensordata[name])

        if name[0] == '_':
            raise AttributeError

        nextview = LView(self._lib, *self._sel, name)
        # if len(nextview._cached) == 0:
        #     raise AttributeError("No distributions matching spec `%s` in library"%str(name))
        return nextview

    def __call__(self, *tags, **kwspec):
        filters = list(self._filters)
        if len(tags) == 1 and hasattr(tags[0], '__call__') and len(self._sel) > 0:
            def interpreted_filter(taglist):
                val = dict(filter(_has2, taglist)).get(self._most_recent_tag, None)
                return tags[0](val) if val is not None else (self._most_recent_tag in taglist)
                # TODO: LOOK UP [self._sel[-1]]
            interpreted_filter.__doc__ = getsource(tags[0])
            filters.append(interpreted_filter)
            T = self._sel - {self._most_recent_tag}
        else:
            T = [*self._sel, *tags]

        nextview = LView(self._lib,  *T, _filters=filters, **kwspec)
        # if len(nextview._cached) == 0:
        #     raise ValueError("No distributions matching spec `%s` in library"%str(name))
        return nextview

    # Before uncommenting: either make underscores special, change
    # the constructor where things are initialized, or enable a flag after
    # construction.
    # def __setattr__(self, key, dist):
    #     self._lib[name, frozenset(*self._sel, key)] = dist

    # This doesn't work.
    # def __set__(self, obj, value):
    #     print("__set__ called with: ", self, obj, value)
    #     self._lib[self._sel] = value
