.. _sh-basis:

========================
Spherical Harmonic bases
========================

Spherical Harmonics (SH) are functions defined on the sphere. A collection of SH
can used as a basis function to represent and reconstruct any function on the
surface of a unit sphere.

Spherical harmonics are ortho-normal functions defined by:

..  math::

    Y_l^m(\theta, \phi) = \sqrt{\frac{2l + 1}{4 \pi} \frac{(l - m)!}{(l + m)!}} P_l^m( cos \theta) e^{i m \phi}

where $l$ is the band index, $m$ is the order, $P_l^m$ is an associated
$l$-th degree, $m$-th order Legendre polynomial, and $(\theta, \phi)$ is the
representation of the direction vector in the spherical coordinate.

A function $f(\theta, \phi)$ can be represented using a spherical harmonics
basis using the spherical harmonics coefficients $a_l^m$, which can be
computed using the expression:

..  math::

    a_l^m = \int_S f(\theta, \phi) Y_l^m(\theta, \phi) ds

Once the coefficients are computed, the function $f(\theta, \phi)$ can be
approximately computed as:

..  math::

    f(\theta, \phi) = \sum_{l = 0}^{\inf} \sum_{m = -l}^{l} a^m_l Y_l^m(\theta, \phi)

In HARDI, the Orientation Distribution Function (ODF) is a function on the
sphere.

Several Spherical Harmonics bases have been proposed in the diffusion imaging
literature for the computation of the ODF. DIPY implements two of these in the
:mod:`~dipy.reconst.shm` module tool set:

- The basis proposed by Descoteaux *et al.* [1]_:

..  math::

    Y_i(\theta, \phi) =
     \begin{cases}
     \sqrt{2} \Re(Y_l^{|m|}(\theta, \phi)) & -l \leq m < 0, \\
     Y_l^0(\theta, \phi) & m = 0, \\
     \sqrt{2} \Im(Y_l^m(\theta, \phi)) & 0 < m \leq l
     \end{cases}

- The basis proposed by Tournier *et al.* [2]_:

..  math::

    Y_i(\theta, \phi) =
     \begin{cases}
     \Im(Y_l^{|m|}(\theta, \phi)) & -l \leq m < 0, \\
     Y_l^0(\theta, \phi) & m = 0, \\
     \Re(Y_{l}^m(\theta, \phi)) & 0 < m \leq l
     \end{cases}

In both cases, $\Re$ denotes the real part of the spherical harmonic basis, and
$\Im$ denotes the imaginary part.

In practice, a maximum even order $k$ is chosen such that $k \leq l$. By only
taking into account even order SH functions, the above bases can be used to
reconstruct symmetric spherical functions. The choice of an even order is
motivated by the symmetry of the diffusion process around the origin.

Both bases are also available as full SH bases, where odd order SH functions are
also taken into account when reconstructing a spherical function. These full
bases can successfully reconstruct asymmetric signals as well as symmetric
signals.

Descoteaux *et al.* [1]_ use the Q-Ball Imaging (QBI) formalization to recover
the ODF, while Tournier *et al.* [2]_ use the Spherical Deconvolution (SD)
framework to recover the ODF.


References
----------
.. [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
       Regularized, Fast, and Robust Analytical Q‐ball Imaging.
       Magn. Reson. Med. 2007;58:497-510.
.. [2] Tournier J.D., Calamante F. and Connelly A. Robust determination
       of the fibre orientation distribution in diffusion MRI:
       Non-negativity constrained super-resolved spherical deconvolution.
       NeuroImage. 2007;35(4):1459–1472.
