import numpy as np
cimport numpy as cnp
cimport cython
import os.path

from dipy.data import get_sphere
from dipy.core.sphere import disperse_charges, Sphere, HemiSphere
from tempfile import gettempdir
from libc.math cimport sqrt, exp, fabs, cos, sin, tan, acos, atan2
from math import ceil

cdef class EnhancementKernel:

    cdef double D33
    cdef double D44
    cdef double t
    cdef int kernelsize
    cdef double kernelmax
    cdef double [:, :] orientations_list
    cdef double [:, :, :, :, :] lookuptable
    cdef object sphere

    def __init__(self, D33, D44, t, force_recompute=False,
                 orientations=None, verbose=True):
        """ Compute a look-up table for the contextual
        enhancement kernel

        Parameters
        ----------
        D33 : float
            Spatial diffusion
        D44 : float
            Angular diffusion
        t : float
            Diffusion time
        force_recompute : boolean
            Always compute the look-up table even if it is available
            in cache. Default is False.
        orientations : integer or Sphere object
            Specify the number of orientations to be used with
            electrostatic repulsion, or provide a Sphere object.
            The default sphere is 'repulsion100'.
        verbose : boolean
            Enable verbose mode.
            
        References
        ----------
        [Meesters2016_ISMRM] S. Meesters, G. Sanguinetti, E. Garyfallidis, 
                             J. Portegies, R. Duits. (2016) Fast implementations 
                             of contextual PDE’s for HARDI data processing in 
                             DIPY. ISMRM 2016 conference.
        [DuitsAndFranken_IJCV] R. Duits and E. Franken (2011) Left-invariant diffusions 
                        on the space of positions and orientations and their 
                        application to crossing-preserving smoothing of HARDI 
                        images. International Journal of Computer Vision, 92:231-264.
        [Portegies2015] J. Portegies, G. Sanguinetti, S. Meesters, and R. Duits.
                        (2015) New Approximation of a Scale Space Kernel on SE(3) 
                        and Applications in Neuroimaging. Fifth International
                        Conference on Scale Space and Variational Methods in
                        Computer Vision
        [Portegies2015b] J. Portegies, R. Fick, G. Sanguinetti, S. Meesters, 
                         G. Girard, and R. Duits. (2015) Improving Fiber 
                         Alignment in HARDI by Combining Contextual PDE flow with 
                         Constrained Spherical Deconvolution. PLoS One.
        """

        # save parameters as class members
        self.D33 = D33
        self.D44 = D44
        self.t = t

        # define a sphere
        if isinstance(orientations, Sphere):
            # use the sphere defined by the user
            sphere = orientations
        elif isinstance(orientations, (int, long, float)):
            # electrostatic repulsion based on number of orientations
            n_pts = int(orientations)
            if n_pts == 0:
                sphere = None
            else:
                theta = np.pi * np.random.rand(n_pts)
                phi = 2 * np.pi * np.random.rand(n_pts)
                hsph_initial = HemiSphere(theta=theta, phi=phi)
                sphere, potential = disperse_charges(hsph_initial, 5000)
        else:
            # use default
            sphere = get_sphere('repulsion100')

        if sphere is not None:
            self.orientations_list = sphere.vertices
            self.sphere = sphere
        else:
            self.orientations_list = np.zeros((0,0))
            self.sphere = None
        
        # file location of the lut table for saving/loading
        kernellutpath = os.path.join(gettempdir(), 
                                     "kernel_d33@%4.2f_d44@%4.2f_t@%4.2f_numverts%d.npy" \
                                       % (D33, D44, t, len(self.orientations_list)))

        # if LUT exists, load
        if not force_recompute and os.path.isfile(kernellutpath):
            if verbose:
                print("The kernel already exists. Loading from " + kernellutpath)
            self.lookuptable = np.load(kernellutpath)

        # else, create
        else:
            if verbose:
                print("The kernel doesn't exist yet. Computing...")
            self.create_lookup_table(verbose)
            if self.sphere is not None:
                np.save(kernellutpath, self.lookuptable)
            
    def get_lookup_table(self):
        """ Return the computed look-up table.
        """
        return self.lookuptable

    def get_orientations(self):
        """ Return the orientations.
        """
        return self.orientations_list
        
    def get_sphere(self):
        """ Get the sphere corresponding with the orientations
        """
        return self.sphere

    def evaluate_kernel(self, x, y, r, v):
        """ Evaluate the kernel at position x relative to
        position y, with orientation r relative to orientation v.

        Parameters
        ----------
        x : 1D ndarray
            Position x
        y : 1D ndarray
            Position y
        r : 1D ndarray
            Orientation r
        v : 1D ndarray
            Orientation v

        Returns
        -------
        kernel_value : double
        """
        return self.k2(x, y, r, v)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void create_lookup_table(self, verbose=True):
        """ Compute the look-up table based on the parameters set
        during class initialization

        Parameters
        ----------
        verbose : boolean
            Enable verbose mode.
        """
        self.estimate_kernel_size(verbose)

        cdef:
            double [:, :] orientations = np.copy(self.orientations_list)
            cnp.npy_intp OR1 = orientations.shape[0]
            cnp.npy_intp OR2 = orientations.shape[0]
            cnp.npy_intp N = self.kernelsize
            cnp.npy_intp hn = (N-1)/2
            cnp.npy_intp angv, angr, xp, yp, zp
            double [:] x
            double [:] y
            cdef double [:, :, :, :, :] lookuptablelocal
            double kmax = self.kernelmax
            double l1norm
            double kernelval

        lookuptablelocal = np.zeros((OR1, OR2, N, N, N))
        x = np.zeros(3)
        y = np.zeros(3) # constant at (0,0,0)

        with nogil:

            for angv in range(OR1):
                for angr in range(OR2):
                    for xp in range(-hn, hn + 1):
                        for yp in range(-hn, hn + 1):
                            for zp in range(-hn, hn + 1):

                                x[0] = xp
                                x[1] = yp
                                x[2] = zp

                                lookuptablelocal[angv,
                                                 angr,
                                                 xp + hn,
                                                 yp + hn,
                                                 zp + hn] = self.k2(x, y, orientations[angr,:], orientations[angv,:])

        # save to class member
        self.lookuptable = lookuptablelocal

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void estimate_kernel_size(self, verbose=True):
        """ Estimates the dimensions the kernel should
        have based on the kernel parameters.

        Parameters
        ----------
        verbose : boolean
            Enable verbose mode.
        """

        cdef:
            double [:] x
            double [:] y
            double [:] r
            double [:] v
            double i

        x = np.array([0., 0., 0.])
        y = np.array([0., 0., 0.])
        r = np.array([0., 0., 1.])
        v = np.array([0., 0., 1.])

        # evaluate at origin
        self.kernelmax = self.k2(x, y, r, v)

        with nogil:

            # determine a good kernel size
            i = 0.0
            while True:
                i += 0.1
                x[2] = i
                kval = self.k2(x, y, r, v) / self.kernelmax
                if(kval < 0.1):
                    break

        N = ceil(i) * 2
        if N % 2 == 0:
            N -= 1

        if verbose:
            print("Dimensions of kernel: %dx%dx%d" % (N, N, N))

        self.kernelsize = N

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    cdef double k2(self, double [:] x, double [:] y,
                   double [:] r, double [:] v) nogil:
        """ Evaluate the kernel at position x relative to
        position y, with orientation r relative to orientation v.

        Parameters
        ----------
        x : 1D ndarray
            Position x
        y : 1D ndarray
            Position y
        r : 1D ndarray
            Orientation r
        v : 1D ndarray
            Orientation v

        Returns
        -------
        kernel_value : double
        """
        cdef:
            double [:] a
            double [:,:] transm
            double [:] arg1
            double [:] arg2p
            double [:] arg2
            double [:] c
            double kernelval

        with gil:

            a = np.subtract(x, y)
            transm = np.transpose(R(euler_angles(v)))
            arg1 = np.dot(transm, a)
            arg2p = np.dot(transm, r)
            arg2 = euler_angles(arg2p)
            c = self.coordinate_map(arg1[0], arg1[1], arg1[2],
                                    arg2[0], arg2[1])
            kernelval = self.kernel(c)

        return kernelval

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double [:] coordinate_map(self, double x, double y,
                                   double z, double beta,
                                   double gamma) nogil:
        """ Compute a coordinate map for the kernel

        Parameters
        ----------
        x : double
            X position
        y : double
            Y position
        z : double
            Z position
        beta : double
            First Euler angle
        gamma : double
            Second Euler angle

        Returns
        -------
        c : 1D ndarray
            array of coordinates for kernel
        """

        cdef:
            double [:] c
            double q
            double cg
            double cotq2

        with gil:

            c = np.zeros(6)

        if beta == 0:
            c[0] = x
            c[1] = y
            c[2] = z
            c[3] = c[4] = c[5] = 0

        else:
            q = fabs(beta)
            cg = cos(gamma)
            sg = sin(gamma)
            cotq2 = 1.0 / tan(q/2)

            c[0] = -0.5*z*beta*cg + \
                    x*(1 - (beta*beta*cg*cg * (1 - 0.5*q*cotq2)) / (q*q)) - \
                    (y*beta*beta*cg*sg * (1 - 0.5*q*cotq2)) / (q*q)
            c[1] = -0.5*z*beta*sg - \
                    (x*beta*beta*cg*sg * (1 - 0.5*q*cotq2)) / (q*q) + \
                    y * (1 - (beta*beta*sg*sg * (1 - 0.5*q*cotq2)) / (q*q))
            c[2] = 0.5*x*beta*cg + 0.5*y*beta*sg + \
                   z * (1 + ((1 - 0.5*q*cotq2) * (-beta*beta*cg*cg - \
                        beta*beta*sg*sg)) / (q*q))
            c[3] = beta * (-sg)
            c[4] = beta * cg
            c[5] = 0

        return c

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double kernel(self, double [:] c) nogil:
        """ Internal function, evaluates the kernel based on the coordinate map.

        Parameters
        ----------
        c : 1D ndarray
            array of coordinates for kernel

        Returns
        -------
        kernel_value : double
        """
        cdef double output = 1 / (8*sqrt(2))
        output *= sqrt(PI)*self.t*sqrt(self.t*self.D33)*sqrt(self.D33*self.D44)
        output *= 1 / (16*PI*PI*self.D33*self.D33*self.D44*self.D44*self.t*self.t*self.t*self.t)
        output *= exp(-sqrt((c[0]*c[0] + c[1]*c[1]) / (self.D33*self.D44) + \
                   (c[2]*c[2] / self.D33 + (c[3]*c[3]+c[4]*c[4]) / self.D44) * \
                   (c[2]*c[2] / self.D33 + (c[3]*c[3]+c[4]*c[4]) / self.D44) + \
                    c[5]*c[5]/self.D44) / (4*self.t))
        return output

cdef double PI = 3.1415926535897932

@cython.wraparound(False)
@cython.boundscheck(False)
cdef double [:] euler_angles(double [:] inp) nogil:
    """ Compute the Euler angles for a given input vector

    Parameters
    ----------
    inp : 1D ndarray
        Input vector

    Returns
    -------
    euler_angles : 1D ndarray
    """
    cdef:
        double x
        double y
        double z
        double [:] output

    x = inp[0]
    y = inp[1]
    z = inp[2]

    with gil:

        output = np.zeros(3)

    # handle the case (0,0,1)
    if x*x < 10e-6 and y*y < 10e-6 and (z-1) * (z-1) < 10e-6:
        output[0] = 0
        output[1] = 0

    # handle the case (0,0,-1)
    elif x*x < 10e-6 and y*y < 10e-6 and (z+1) * (z+1) < 10e-6:
        output[0] = PI
        output[1] = 0

    # all other cases
    else:
        output[0] = acos(z)
        output[1] = atan2(y, x)

    return output

@cython.wraparound(False)
@cython.boundscheck(False)
cdef double [:,:] R(double [:] inp) nogil:
    """ Compute the Rotation matrix for a given input vector

    Parameters
    ----------
    inp : 1D ndarray
        Input vector

    Returns
    -------
    rotation_matrix : 2D ndarray
    """
    cdef:
        double beta
        double gamma
        double [:] output
        double cb
        double sb
        double cg
        double sg

    beta = inp[0]
    gamma = inp[1]
    
    with gil:
        
        output = np.zeros(9)

    cb = cos(beta)
    sb = sin(beta)
    cg = cos(gamma)
    sg = sin(gamma)

    output[0] = cb * cg
    output[1] = -sg
    output[2] = cg * sb
    output[3] = cb * sg
    output[4] = cg
    output[5] = sb * sg
    output[6] = -sb
    output[7] = 0
    output[8] = cb

    with gil:

        return np.reshape(output, (3,3))
