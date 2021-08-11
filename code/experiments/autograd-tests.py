#%load_ext autoreload
#%autoreload 2
#%cd code/

from autograd import grad, jacobian, elementwise_grad as egrad
import primitives as P
from environs import Erand as E

import autograd.numpy as np
#import numpy as np 

R = np.zeros(E.Sshape)
R[1] = 1

jacobian(P.fwd, 2)(E,R, 0.9)
jacobian(P.fwd, 1)(E,R, 0.9, temp=0.1)
P.fwd(E,R, 0.9, temp=0.1)

jacobian(P.value_iter, 1)(E,R, 0.9)
_.shape




P.value_iter(E,R, 0.9)
R
