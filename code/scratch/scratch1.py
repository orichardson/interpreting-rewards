
# %cd code
# %% a
from dist import RawJointDist as RJD
RJD([3/8, 3/8, 1/4]).H(...)
RJD([4/8, 2/8, 1/4]).H(...)

# %%
from primitives import *

E3 = { 's0' : { 'left' : 's0', 'right' : 's1'},
        's1' : {'left' : 's1'}}
        
        
adjust_both(a, π, ρ, γ)

from collections import defaultdict

d = defaultdict(list)

d[0] += [1]
d
