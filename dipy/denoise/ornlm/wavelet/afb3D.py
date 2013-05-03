import ornlm_module
import numpy as np
from cshift3D import cshift3D
def permutationInverse(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

def afb3D_A(x, af, d):
    lpf=af[:, 0]
    hpf=af[:, 1]
    # permute dimensions of x so that dimension d is first.
    p=[(i+d)%3 for i in xrange(3)]
    x=x.transpose(p)
    # filter along dimension 0
    (N1, N2, N3)=x.shape
    L=af.shape[0]//2
    x=cshift3D(x, -L, 0);
    n1Half=N1//2
    lo=np.zeros((L+n1Half, N2, N3));
    hi=np.zeros((L+n1Half, N2, N3));
    for k in xrange(N3):
        lo[:, :, k]=ornlm_module.firdnpy(x[:, :, k], lpf);
    lo[:L]=lo[:L]+lo[n1Half:n1Half+L, :, :];
    lo=lo[:n1Half, :, :];
    
    for k in xrange(N3):
        hi[:, :, k]=ornlm_module.firdnpy(x[:, :, k], hpf);
    hi[:L]=hi[:L]+hi[n1Half:n1Half+L, :, :];
    hi=hi[:n1Half, :, :];
    #permute dimensions of x (inverse permutation)
    q=permutationInverse(p)
    lo=lo.transpose(q)
    hi=hi.transpose(q)
    return lo, hi



def afb3D(x, af1, af2=None, af3=None):
    if af2==None:
        af2=af1
    if af3==None:
        af3=af1
    # filter along dimension 0
    L,H=afb3D_A(x, af1, 0);
    # filter along dimension 1
    LL,LH=afb3D_A(L, af2, 1);
    HL,HH=afb3D_A(H, af2, 1);
    # filter along dimension 3
    LLL,LLH=afb3D_A(LL, af3, 2);
    LHL,LHH=afb3D_A(LH, af3, 2);
    HLL,HLH=afb3D_A(HL, af3, 2);
    HHL,HHH=afb3D_A(HH, af3, 2);
    return LLL, [LLH, LHL, LHH, HLL, HLH, HHL, HHH]
