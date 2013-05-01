import ornlm_module
import numpy as np
from cshift3D import cshift3D
def permutationInverse(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

def sfb3D_A(lo, hi, sf, d):
    lpf=sf[:, 0]
    hpf=sf[:, 1]
    # permute dimensions of lo and hi so that dimension d is first.
    p=[(i+d)%3 for i in xrange(3)]
    lo=lo.transpose(p)
    hi=hi.transpose(p)

    (N1, N2, N3)=lo.shape
    N=2*N1
    L=sf.shape[0]
    y=np.zeros((N+L-2, N2, N3))
    for k in xrange(N3):
        y[:, :, k] = np.array(ornlm_module.upfirpy(lo[:, :, k], lpf)) + np.array(ornlm_module.upfirpy(hi[:, :, k], hpf))
    y[:(L-2), :, :] = y[:(L-2), :, :] + y[N:(N+L-2), :, :]
    y=y[:N, :, :]
    y=cshift3D(y, 1-L/2, 0);
    #permute dimensions of y (inverse permutation)
    q=permutationInverse(p)
    y=y.transpose(q);
    return y

def sfb3D(lo, hi, sf1, sf2=None, sf3=None):
    if sf2==None:
        sf2=sf1
    if sf3==None:
        sf3=sf1
    LLL=lo;
    LLH=hi[0];
    LHL=hi[1];
    LHH=hi[2];
    HLL=hi[3];
    HLH=hi[4];
    HHL=hi[5];
    HHH=hi[6];
    # filter along dimension 2
    LL=sfb3D_A(LLL, LLH, sf3, 2);
    LH=sfb3D_A(LHL, LHH, sf3, 2);
    HL=sfb3D_A(HLL, HLH, sf3, 2);
    HH=sfb3D_A(HHL, HHH, sf3, 2);
    # filter along dimension 1
    L=sfb3D_A(LL, LH, sf2, 1);
    H=sfb3D_A(HL, HH, sf2, 1);
    # filter along dimension 0
    y=sfb3D_A(L, H, sf1, 0);
    return y


