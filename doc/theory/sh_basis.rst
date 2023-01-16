.. _sh-basis:

========================
Spherical Harmonic bases
========================

Spherical Harmonics (SH) are functions defined on the sphere. A collection of
SH can be used as a basis function to represent and reconstruct any function on
the surface of a unit sphere.

Spherical harmonics are orthonormal functions defined by:

..  math::

    Y_l^m(\theta, \phi) = \sqrt{\frac{2l + 1}{4 \pi} \frac{(l - m)!}{(l + m)!}} P_l^m( cos \theta) e^{i m \phi}

where $l$ is the order, $m$ is the degree, $P_l^m$ is an associated
$l$-th order, $m$-th degree Legendre polynomial, and $(\theta, \phi)$ is the
representation of the direction vector in spherical coordinates. The relation
between $Y_l^{m}$ and $Y_l^{-m}$ is given by:

..  math::

    Y_l^{-m}(\theta, \phi) = (-1)^m \overline{Y_l^m}

where $\overline{Y_l^m}$ is the complex conjugate of $Y_l^m$ defined as
$\overline{Y_l^m} = \Re(Y_l^m) - \Im(Y_l^m)$.

A function $f(\theta, \phi)$ can be represented using a spherical harmonics
basis using the spherical harmonics coefficients $a_l^m$, which can be
computed using the expression:

..  math::

    a_l^m = \int_S f(\theta, \phi) Y_l^m(\theta, \phi) ds

Once the coefficients are computed, the function $f(\theta, \phi)$ can be
computed as:

..  math::

    f(\theta, \phi) = \sum_{l = 0}^{\infty} \sum_{m = -l}^{l} a^m_l Y_l^m(\theta, \phi)

In HARDI, the Orientation Distribution Function (ODF) is a function on the
sphere. Therefore, SH functions offer the ideal framework for reconstructing
the ODF. Descoteaux *et al.* [1]_ use the Q-Ball Imaging (QBI) formalization
to recover the ODF, while Tournier *et al.* [2]_ use the Spherical Deconvolution
(SD) framework.

Several modified SH bases have been proposed in the diffusion imaging literature
for the computation of the ODF. DIPY implements two of these in the 
:mod:`~dipy.reconst.shm` module. Below are the formal definitions taken
directly from the literature.

- The basis proposed by Descoteaux *et al.* [1]_:

..  math::

    Y_i(\theta, \phi) =
     \begin{cases}
     \sqrt{2} \Re(Y_l^{m}(\theta, \phi)) & -l \leq m < 0, \\
     Y_l^0(\theta, \phi) & m = 0, \\
     \sqrt{2} \Im(Y_l^m(\theta, \phi)) & 0 < m \leq l
     \end{cases}

- The basis proposed by Tournier *et al.* [2]_:

..  math::

    Y_i(\theta, \phi) =
     \begin{cases}
     \Im(Y_l^{m}(\theta, \phi)) & -l \leq m < 0, \\
     Y_l^0(\theta, \phi) & m = 0, \\
     \Re(Y_{l}^m(\theta, \phi)) & 0 < m \leq l
     \end{cases}

In both cases, $\Re$ denotes the real part of the spherical harmonic basis, and
$\Im$ denotes the imaginary part. The SH bases are both orthogonal and real. Moreover,
the `descoteaux07` basis is orthonormal.

In both cases, $\Re$ denotes the real part of the SH basis, and $\Im$ denotes
the imaginary part. By alternately selecting the real or imaginary part of the
original SH basis, the modified SH bases have the properties of being both
orthogonal and real. Moreover, due to the presence of the $\sqrt{2}$ factor,
the basis proposed by Descoteaux *et al.* is orthonormal.

The SH bases implemented in DIPY for versions 1.2 and below differ slightly
from the literature. Their implementation is given below.

- The ``descoteaux07`` basis is based on the one proposed by Descoteaux *et al.*
  [1]_ and is given by:

..  math::

    Y_i(\theta, \phi) =
     \begin{cases}
     \sqrt{2} \Re(Y_l^{|m|}(\theta, \phi)) & -l \leq m < 0, \\
     Y_l^0(\theta, \phi) & m = 0, \\
     \sqrt{2} \Im(Y_l^m(\theta, \phi)) & 0 < m \leq l
     \end{cases}

- The ``tournier07`` basis is based on the one proposed by Tournier *et al.*
  [2]_ and is given by:

..  math::

    Y_i(\theta, \phi) =
     \begin{cases}
     \Im(Y_l^{|m|}(\theta, \phi)) & -l \leq m < 0, \\
     Y_l^0(\theta, \phi) & m = 0, \\
     \Re(Y_{l}^m(\theta, \phi)) & 0 < m \leq l
     \end{cases}

These bases differ from the literature by the presence of an absolute value around
$m$ when $m < 0$. Due to relations $-p = |p| ; \forall p < 0$ and
$Y_l^{-m}(\theta, \phi) = (-1)^m \overline{Y_l^m}$, the effect of this change is a
sign flip for the SH functions of even degree $m < 0$. This has no effect on the
mathematical properties of each basis.

The ``tournier07`` SH basis defined above is the basis used in MRtrix 0.2 [3]_.
However, the omission of the $\sqrt{2}$ factor seen in the basis from Descoteaux
*et al.* [1]_ makes it non-orthonormal. For this reason, the MRtrix3 [4]_ SH
basis uses a new basis including the normalization factor.

Since DIPY 1.3, the ``descoteaux07`` and ``tournier07`` SH bases have been
updated in order to agree with the literature and the latest MRtrix3
implementation. While previous bases are still available as *legacy* bases, 
the ``descoteaux07`` and ``tournier07`` bases now default to:

..  math::

    Y_i(\theta, \phi) =
     \begin{cases}
     \sqrt{2} \Re(Y_l^{m}(\theta, \phi)) & -l \leq m < 0, \\
     Y_l^0(\theta, \phi) & m = 0, \\
     \sqrt{2} \Im(Y_l^m(\theta, \phi)) & 0 < m \leq l
     \end{cases}

for the ``descoteaux07`` basis and

..  math::

    Y_i(\theta, \phi) =
     \begin{cases}
     \sqrt{2} \Im(Y_l^{|m|}(\theta, \phi)) & -l \leq m < 0, \\
     Y_l^0(\theta, \phi) & m = 0, \\
     \sqrt{2} \Re(Y_{l}^m(\theta, \phi)) & 0 < m \leq l
     \end{cases}

for the ``tournier07`` basis. Both bases are very similar, with their only
difference being the sign of $m$ for which the imaginary and real parts of
the spherical harmonic $Y_{l}^m$ are used.

In practice, a maximum order $k$ is used to truncate the SH series. By
only taking into account even order SH functions, the above bases can be used
to reconstruct symmetric spherical functions. The choice of an even order is
motivated by the symmetry of the diffusion process around the origin.

Both bases are also available as full SH bases, where odd order SH functions
are also taken into account when reconstructing a spherical function. These
full bases can successfully reconstruct asymmetric signals as well as
symmetric signals.

References
----------
.. [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
       Regularized, Fast, and Robust Analytical Q‐ball Imaging.
       Magn. Reson. Med. 2007;58:497-510.
.. [2] Tournier J.D., Calamante F. and Connelly A. Robust determination
       of the fibre orientation distribution in diffusion MRI:
       Non-negativity constrained super-resolved spherical deconvolution.
       NeuroImage. 2007;35(4):1459–1472.
.. [3] Tournier J-D, Smith R, Raffelt D, Tabbara R, Dhollander T,
       Pietsch M, et al. MRtrix3: A fast, flexible and open software
       framework for medical image processing and visualisation.
       NeuroImage. 2019 Nov 15;202:116-137.
.. [4] Tournier J-D, Smith R, Raffelt D, Tabbara R, Dhollander T,
       Pietsch M, et al. MRtrix3: A fast, flexible and open software
       framework for medical image processing and visualisation.
       NeuroImage. 2019 Nov 15;202:116-137.
