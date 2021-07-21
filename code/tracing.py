from collections import defaultdict
from utils import matches_any, dictwo
from functools import wraps

import re

class TraceStore:
    def __init__(self, what_to_store = None):
        self.trace = defaultdict(list)
        self._data_list = []
        self.what_to_store = what_to_store
        self.prefix = ""
        self.suffix = ""


    def __call__(self, **kw_data):
        for k,v in kw_data.items():
            if matches_any(self.what_to_store, k):
                tag = self.prefix + k + self.suffix
                self._data_list.append((v, dict(name=k,prefix=self.prefix,suffix=self.suffix,tag=tag)))
                self.trace[tag].append(v)
                
    def copy(self, **kwargs):
        """ shallow copy;
        shares an underlying store, but the reference might add things to it differently"""
        t = TraceStore(self.what_to_store)
        t.trace = self.trace
        t._data_list = self._data_list
        t.prefix = self.prefix
        t.suffix = self.suffix
        for k,v in kwargs.items():
            setattr(t, k, v)
        
        return t
        
    def matching(self, regexstr):
        pattern = re.compile(regexstr)
        return {k:v for k,v in self.trace.items() if pattern.search(k)}
        
    def firstN(self, N, pattern=re.compile("")):
        rslt = defaultdict(list)

        for v,d in self._data_list:
            if len(rslt) >= N:
                break
            if pattern.search(d['tag']):
                rslt[d['tag']] = v

        return rslt
        # pass
        
    def __bool__(self):
        return bool(self.what_to_store)
                
    def __getitem__(self, key):
        return self.trace[key]
        
    def __radd__(self, prefix):
        return self.copy(prefix = prefix + self.prefix)
    def __add__(self, suffix):
        return self.copy(suffix = self.suffix + suffix)
        
    # not really that helpful but exposes the "*" to the API.
    @staticmethod
    def all():
        return TraceStore("*")

none = TraceStore(False)
        
def with_tracing( func ):
    @wraps(func)
    def trace_wrapper(*args, **kwargs):
        if "trace" in kwargs:
            descr = kwargs['trace']
            return func(*args, **dictwo(kwargs,["trace"]))
            
        return func(*args, **kwargs) 
        
    return trace_wrapper
