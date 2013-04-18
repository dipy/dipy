""" Random number generation utilities """
from __future__ import division, print_function, absolute_import

from math import floor
from platform import architecture

def WichmannHill2006():
    '''
    B.A. Wichmann, I.D. Hill, Generating good pseudo-random numbers,
    Computational Statistics & Data Analysis, Volume 51, Issue 3, 1
    December 2006, Pages 1614-1622, ISSN 0167-9473, DOI:
    10.1016/j.csda.2006.05.019. (http://www.sciencedirect.com/science/article/B6V8V-4K7F86W-2/2/a3a33291b8264e4c882a8f21b6e43351)
    for advice on generating many sequences for use together, and on alternative algorithms and codes

    Examples
    ----------
    >>> from dipy.core import rng
    >>> rng.ix, rng.iy, rng.iz, rng.it = 100001, 200002, 300003, 400004
    >>> N = 1000
    >>> a = [rng.WichmannHill2006() for i in range(N)]
    '''

    global ix, iy, iz, it

    if architecture()[0] == '64':

        #If 64 bits are available then the following lines of code will be faster.
        ix = (11600 * ix) % 2147483579
        iy = (47003 * iy) % 2147483543
        iz = (23000 * iz) % 2147483423
        it = (33000 * it) % 2147483123

    else:

        #If only 32 bits are available

        ix = 11600 * (ix % 185127) - 10379 * (ix / 185127)
        iy = 47003 * (ix %  45688) - 10479 * (iy /  45688)
        iz = 23000 * (iz %  93368) - 19423 * (iz /  93368)
        it = 33000 * (it %  65075) -  8123 * (it /  65075)

        if ix < 0:
            ix = ix + 2147483579
        if iy < 0:
            iy = iy + 2147483543
        if iz < 0:
            iz = iz + 2147483423
        if it < 0:
            it = it + 2147483123

    W = ix/2147483579.0 + iy/2147483543.0 + iz/2147483423.0 + it/2147483123.0

    return W - floor(W)


def WichmannHill1982():
    '''
    Algorithm AS 183 Appl. Statist. (1982) vol.31, no.2

    Returns a pseudo-random number rectangularly distributed
    between 0 and 1.   The cycle length is 6.95E+12 (See page 123
    of Applied Statistics (1984) vol.33), not as claimed in the
    original article.

    ix, iy and iz should be set to integer values between 1 and
    30000 before the first entry.

    Integer arithmetic up to 5212632 is required.
    '''

    import numpy as np

    global ix, iy, iz

    ix = (171 * ix) % 30269
    iy = (172 * iy) % 30307
    iz = (170 * iz) % 30323

    '''
    If integer arithmetic only up to 30323 (!) is available, the preceding
    3 statements may be replaced by:

    ix = 171 * (ix % 177) -  2 * (ix / 177)
    iy = 172 * (iy % 176) - 35 * (iy / 176)
    iz = 170 * (iz % 178) - 63 * (iz / 178)

    if ix < 0:
        ix = ix + 30269
    if iy < 0:
        iy = iy + 30307
    if iz < 0:
        iz = iz + 30323
    '''
    return np.remainder(np.float(ix) / 30269. + np.float(iy) / 30307.
                          + np.float(iz) / 30323., 1.0)


def LEcuyer():
    '''
    Generate uniformly distributed random numbers using the 32-bit
    generator from figure 3 of:
        L'Ecuyer, P. Efficient and portable combined random number
        generators, C.A.C.M., vol. 31, 742-749 & 774-?, June 1988.

    The cycle length is claimed to be 2.30584E+18
    '''

    global s1, s2

    k = s1 / 53668
    s1 = 40014 * (s1 - k * 53668) - k * 12211
    if s1 < 0:
        s1 = s1 + 2147483563
    k = s2 / 52774
    s2 = 40692 * (s2 - k * 52774) - k * 3791
    if  s2 < 0:
        s2 = s2 + 2147483399
    z = s1 - s2
    if z < 0:
        z = z + 2147483562

    return z / 2147483563.
