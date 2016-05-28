import numpy as np
from dipy.denoise import ornlm

'''
 Functions for Wavelet Transforms in 3D domain

 Code adapted from

 WAVELET SOFTWARE AT POLYTECHNIC UNIVERSITY, BROOKLYN, NY
 http://taco.poly.edu/WaveletSoftware/
'''


def cshift3D(x, m, d):
    '''
    3D Circular Shift
    
    USAGE:
       y = cshift3D(x, m, d)
    INPUT:
       x - N1 by N2 by N3 array
       m - amount of shift
       d - dimension of shift (d = 1,2,3)
    OUTPUT:
       y - array x will be shifed by m samples down
           along dimension d

    '''
    s = x.shape
    idx = (np.array(range(s[d])) + (s[d] - m % s[d])) % s[d]
    if d == 0:
        return x[idx, :, :]
    elif d == 1:
        return x[:, idx, :]
    else:
        return x[:, :, idx]


def permutationInverse(perm):
    '''
    Function generating inverse of the permutation
    '''

    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


def afb3D_A(x, af, d):
    
    '''
    3D Analysis Filter Bank
     (along one dimension only)
    
     [lo, hi] = afb3D_A(x, af, d);
    INPUT:
        x - N1xN2xN2 matrix, where min(N1,N2,N3) > 2*length(filter)
               (Ni are even)
        af - analysis filter for the columns
        af(:, 1) - lowpass filter
        af(:, 2) - highpass filter
        d - dimension of filtering (d = 1, 2 or 3)
    OUTPUT:
         lo, hi - lowpass, highpass subbands
    '''

    lpf = af[:, 0]
    hpf = af[:, 1]
    # permute dimensions of x so that dimension d is first.
    p = [(i + d) % 3 for i in xrange(3)]
    x = x.transpose(p)
    # filter along dimension 0
    (N1, N2, N3) = x.shape
    L = af.shape[0] // 2
    x = cshift3D(x, -L, 0)
    n1Half = N1 // 2
    lo = np.zeros((L + n1Half, N2, N3))
    hi = np.zeros((L + n1Half, N2, N3))
    for k in xrange(N3):
        lo[:, :, k] = ornlm.firdn(x[:, :, k], lpf)
    lo[:L] = lo[:L] + lo[n1Half:n1Half + L, :, :]
    lo = lo[:n1Half, :, :]

    for k in xrange(N3):
        hi[:, :, k] = ornlm.firdn(x[:, :, k], hpf)
    hi[:L] = hi[:L] + hi[n1Half:n1Half + L, :, :]
    hi = hi[:n1Half, :, :]
    # permute dimensions of x (inverse permutation)
    q = permutationInverse(p)
    lo = lo.transpose(q)
    hi = hi.transpose(q)
    return lo, hi


def sfb3D_A(lo, hi, sf, d):
    '''
    3D Synthesis Filter Bank
     (along single dimension only)
    
     y = sfb3D_A(lo, hi, sf, d);
     sf - synthesis filters
     d  - dimension of filtering

    '''
    lpf = sf[:, 0]
    hpf = sf[:, 1]
    # permute dimensions of lo and hi so that dimension d is first.
    p = [(i + d) % 3 for i in xrange(3)]
    lo = lo.transpose(p)
    hi = hi.transpose(p)

    (N1, N2, N3) = lo.shape
    N = 2 * N1
    L = sf.shape[0]
    y = np.zeros((N + L - 2, N2, N3))
    for k in xrange(N3):
        y[:, :, k] = (np.array(ornlm.upfir(lo[:, :, k], lpf)) +
                      np.array(ornlm.upfir(hi[:, :, k], hpf)))
    y[:(L - 2), :, :] = y[:(L - 2), :, :] + y[N:(N + L - 2), :, :]
    y = y[:N, :, :]
    y = cshift3D(y, 1 - L / 2, 0)
    # permute dimensions of y (inverse permutation)
    q = permutationInverse(p)
    y = y.transpose(q)
    return y


def sfb3D(lo, hi, sf1, sf2=None, sf3=None):
    '''
    3D Synthesis Filter Bank
    
    USAGE:
       y = sfb3D(lo, hi, sf1, sf2, sf3);
    INPUT:
       lo, hi - lowpass subbands
       sfi - synthesis filters for dimension i
    OUPUT:
       y - output array

    '''
    if sf2 is None:
        sf2 = sf1
    if sf3 is None:
        sf3 = sf1
    LLL = lo
    LLH = hi[0]
    LHL = hi[1]
    LHH = hi[2]
    HLL = hi[3]
    HLH = hi[4]
    HHL = hi[5]
    HHH = hi[6]
    # filter along dimension 2
    LL = sfb3D_A(LLL, LLH, sf3, 2)
    LH = sfb3D_A(LHL, LHH, sf3, 2)
    HL = sfb3D_A(HLL, HLH, sf3, 2)
    HH = sfb3D_A(HHL, HHH, sf3, 2)
    # filter along dimension 1
    L = sfb3D_A(LL, LH, sf2, 1)
    H = sfb3D_A(HL, HH, sf2, 1)
    # filter along dimension 0
    y = sfb3D_A(L, H, sf1, 0)
    return y


def afb3D(x, af1, af2=None, af3=None):
    '''
    3D Analysis Filter Bank
    USAGE:
        [lo, hi] = afb3D(x, af1, af2, af3);
    INPUT:
        x - N1 by N2 by N3 array matrix, where
            1) N1, N2, N3 all even
            2) N1 >= 2*length(af1)
            3) N2 >= 2*length(af2)
            4) N3 >= 2*length(af3)
        afi - analysis filters for dimension i
           afi(:, 1) - lowpass filter
           afi(:, 2) - highpass filter
    OUTPUT:
        lo - lowpass subband
        hi[d], d = 1..7 - highpass subbands
    '''

    if af2 is None:
        af2 = af1
    if af3 is None:
        af3 = af1
    # filter along dimension 0
    L, H = afb3D_A(x, af1, 0)
    # filter along dimension 1
    LL, LH = afb3D_A(L, af2, 1)
    HL, HH = afb3D_A(H, af2, 1)
    # filter along dimension 3
    LLL, LLH = afb3D_A(LL, af3, 2)
    LHL, LHH = afb3D_A(LH, af3, 2)
    HLL, HLH = afb3D_A(HL, af3, 2)
    HHL, HHH = afb3D_A(HH, af3, 2)
    return LLL, [LLH, LHL, LHH, HLL, HLH, HHL, HHH]


def dwt3D(x, J, af):
    '''
    3-D Discrete Wavelet Transform
    
    USAGE:
       w = dwt3D(x, stages, af)
    INPUT:
       x - N1 by N2 by N3 matrix
           1) Ni all even
           2) min(Ni) >= 2^(J-1)*length(af)
       J - number of stages
       af  - analysis filters
    OUTPUT:
       w - cell array of wavelet coefficients

    '''

    w = [None] * (J + 1)
    for k in xrange(J):
        x, w[k] = afb3D(x, af, af, af)
    w[J] = x
    return w


def idwt3D(w, J, sf):
    '''
    Inverse 3-D Discrete Wavelet Transform
    
    USAGE:
       y = idwt3D(w, J, sf)
    INPUT:
       w - wavelet coefficient
       J  - number of stages
       sf - synthesis filters
    OUTPUT:
       y - output array

    '''

    y = w[J]
    for k in range(J)[::-1]:
        y = sfb3D(y, w[k], sf, sf, sf)
    return y
