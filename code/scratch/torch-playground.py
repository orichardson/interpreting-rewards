# %cd code
# %load_ext autoreload
# %autoreload 2

import torch
torch.__version__

torch.functional.logsumexp
torch.prod(torch.tensor([1,1,2]))


torch.logsumexp(torch.tensor([[0,1.],[2,3]]), 1,keepdim=True)
torch.tensor([0,1]) == torch.tensor([0,1])


tns = torch.tensor
X =  tns([0.,1,2, -1, 2], requires_grad=True)
torch.where( X == 1, X , 1-X)


hasattr(X, 'numpy')

( X + tns([2,2,3.]) ).requires_grad


import primitives as P

X.requires_grad
A = P.t_argmax(X, axis=0, temp=0.01)
A.requires_grad
A

am = torch.amax(X, keepdim=True)

am.backward()
X.grad

(X == torch.amax(X,keepdim=True)).requires_grad
X.detach()
