""" Q-ball imaging
Spherical Harmonics
http://www.sjbrown.co.uk/?article=sharmonics
"""


import numpy
import math

factorial = lambda n:reduce(lambda a,b:a*(b+1),range(n),1)

def evaluate_SH(angles, degree, dl=1):
    theta = angles[0]
    phi = angles[1]

    if (dl==2):
        coeff_length = (degree+1)*(degree+2)/2
        B = numpy.zeros([1,coeff_length])
        Btheta = numpy.zeros([1,coeff_length])
        Bphi = numpy.zeros([1,coeff_length])
    elif (dl==1):
        coeff_length = (degree+1)*(degree+1)
        B = numpy.zeros([1,coeff_length])
        Btheta = numpy.zeros([1,coeff_length])
        Bphi = numpy.zeros([1,coeff_length])

    for l in range(0,degree+1,dl):
        if (dl==2):
            center = (l+1)*(l+2)/2 - l
        elif (dl==1):
            center = (l+1)*(l+1) - l
        lconstant = math.sqrt((2*l + 1)/(4*math.pi))
        center = center - 1

        Plm,dPlm = P(l,0,theta)
        B[0,center] = lconstant*Plm
        Btheta[0,center] = lconstant * dPlm
        Bphi[0,center] = 0
        for m in range(1,l+1):
            precoeff = lconstant * math.sqrt(2.0)*math.sqrt(factorial(l - m)/(factorial(l + m)*1.0))
            if (m % 2 == 1):
                precoeff = -precoeff
            Plm,dPlm = P(l,m,theta)
            pre1 = precoeff*Plm
            pre2 = precoeff*dPlm
            B[0,center + m] = pre1*math.cos(m*phi)
            B[0,center - m] = pre1*math.sin(m*phi)
            Btheta[0,center+m] = pre2*math.cos(m*phi)
            Btheta[0,center-m] = pre2*math.sin(m*phi)
            Bphi[0,center+m] = -m*B[0,center-m]
            Bphi[0,center-m] = m*B[0,center+m]
    return B,Btheta,Bphi

def real_spherical_harmonics(angles, coeff, degree, dl=1):
    """
    Given a real-valued spherical function represented by spherical harmonics coefficients,
    this function evalutes its value and gradient at given spherical angles

    SYNTAX: [f, g] = real_spherical_harmonics(angles, coeff, degree, dl);

    INPUTS:
     angles                - [theta,phi] are colatitude and longitude, respectively
     coeff                 - real valued coefficients [a_00, a_1-1, a_10, a_11, ... ]
     degree                - maximum degree of spherical harmonics ;
     dl                    - {1} for full band; 2 for even order only

    OUTPUTS:
     f                     - Evaluated function value f = \sum a_lm*Y_lm
     g                     - derivatives with respect to theta and phi
    """
    B,Btheta,Bphi = evaluate_SH(angles, degree, dl)
    f = sum(-numpy.dot(B,coeff))
    g = numpy.array((-sum(numpy.dot(Btheta,coeff)), -sum(numpy.dot(Bphi,coeff))))
    return f,g



def P(l,m,theta):
    """
    The Legendre polynomials are defined recursively
    """
    pmm = 1
    dpmm = 0
    x = math.cos(theta)
    somx2 = math.sin(theta)
    fact = 1.0

    for i in range(1,m+1):
        dpmm = -fact * (x*pmm + somx2*dpmm)
        pmm = pmm*(-fact * somx2)
        fact = fact+2

    # No need to go any further, rule 2 is satisfied
    if (l == m):
        Plm = pmm
        dPlm = dpmm
        return Plm,dPlm


    # Rule 3, use result of P(m,m) to calculate P(m,m+1)
    pmmp1 = x * (2 * m + 1) * pmm
    dpmmp1 = (2*m+1)*(x*dpmm - somx2*pmm)

    # Is rule 3 satisfied?
    if (l == m + 1):
        Plm = pmmp1
        dPlm = dpmmp1
        return Plm, dPlm

    # Finally, use rule 1 to calculate any remaining cases
    pll = 0
    dpll = 0
    for ll in range(m + 2,l+1):
        # Use result of two previous bands
        pll = (x * (2.0 * ll - 1.0) * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        dpll = ((2.0*ll-1.0)*( x*dpmmp1 - somx2*pmmp1 ) - (ll+m-1.0)*dpmm) / (ll - m)
        # Shift the previous two bands up
        pmm = pmmp1
        dpmm = dpmmp1
        pmmp1 = pll
        dpmmp1 = dpll

    Plm = pll
    dPlm = dpll
    return Plm,dPlm




def cart2sph(x,y,z):
    r = math.sqrt(x**2+y**2+z**2)
    elev = math.atan2(z,math.sqrt(x**2+y**2))
    az = math.atan2(y,x)
    return az, elev, r


def sph2cart(angle):
    # angle  - [theta,phi] colatitude and longitude
    cos = math.cos
    sin = math.sin
    theta,phi = angle
    z = cos(theta)
    x = sin(theta)*cos(phi)
    y = sin(theta)*sin(phi)
    return x,y,z

def construct_SH_basis(pts, degree):
    sph_coord = numpy.array([cart2sph(x,y,z) for x,y,z in pts])
    B  = [spherical_harmonics.evaluate_SH((math.pi/2 - elev,az),degree,2)[0] for az,elev,r in sph_coord]
    # NOTE: spherical harmonics' convention: PHI - azimuth, THETA - polar angle
    # PI/2 - elev is to convert elevation [-PI/2, PI/2] to polar angle [0, PI]
    B = numpy.array(B).squeeze()
    return B
