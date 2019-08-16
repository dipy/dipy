.. _b-and-q:

=========================
 DIY Stuff about b and q
=========================

This is a short note to explain the nature of the ``B_matrix`` found in the
Siemens private (CSA) fields of the DICOM headers of a diffusion weighted
acquisition.  We trying to explain the relationship between the ``B_matrix`` and
the *b value* and the *gradient vector*.  The acquisition is made with a planned
(requested) $b$-value - say $b_{req} = 1000$, and with a requested gradient
direction $\mathbf{g}_{req} = [g_x, g_y, g_z]$ (supposedly a unit vector) and
peak amplitude $G$. When the sequence runs the gradient is modulated by an
amplitude envelope $\rho(t)$ with $\max |\rho(t)| = 1$ so that the time course
of the gradient is $G\rho(t)\mathbf{g}.$ $G$ is measured in units of $T
\mathrm{mm}^-1.$ This leads to an important temporal weighting parameter of the
acquisition:

..  math::

   R = \int_0^T ( \int_0^t \rho ( \tau ) \, d{ \tau } )^2 \, d{t}.

(See Basser, Matiello and LeBihan, 1994.) Another formulation involves
the introduction of k-space. In standard in-plane MR image encoding

.. math::

   \mathbf{k} = \gamma \int \mathbf{g}(t)dt.


For the classical Stejskal and Tanner pulsed gradient spin echo (PGSE)
paradigm where two rectangular pulses of width $\delta$ seconds are
spaced with their onsets $\Delta$ seconds apart $R = \Delta
(\Delta-\delta/3)^2.$ The units of $R$ are $s^3$. The $b$-matrix has
entries

.. math::

   b_{ij} = \gamma^2 G^2 g_i g_j R, 

where $\gamma$ is the gyromagnetic radius (units
$\mathrm{radians}.\mathrm{seconds}^{-1}.T^{-1}$) and $i$ and $j$ are
axis direcrtions from $x,y,z$ . The units of the B-matrix are
$\mathrm{radians}^2 . \mathrm{seconds} .  \mathrm{mm}^{-2}.$

.. math::

   \mathbf{B} = \gamma^2 G^2 R \mathbf{g} \mathbf{g}^T. 

The b-value for the acquisition is the trace of $\mathbf{B}$ and is
given by

.. math::

   b = \gamma^2 G^2 R \|\mathbf{g}\|^2 = \gamma^2 G^2 R.
   
================================
 The B matrix and Siemens DICOM
================================

Though the Stejskal and Tanner formula is available for the classic
PGSE sequence, a different sequence may be used (e.g. TRSE on Siemens
Trio), and anyway the ramps up and down on the gradient field will not
be rectangular. The Siemens scanner software calculates the actual
values of the $b_{ij}$ by numerical integration of the formula above
for $R$. These values are in the form of the 6 'B-matrix' values
$[b_{xx}, b_{xy}, b_{xz}, b_{yy}, b_{yz}, b_{zz}]$.

In this form they are suitable for use in a least squares estimation of
the diffusion tensor via the equations across the set of acquisitions:

.. math::

   \log(A(\mathbf{q})/A(0)) = -(b_{xx}D_{xx} + 2b_{xy}D_{xy} + 2b_{xz}D_{xz} + \
      b_{yy}D_{yy} + 2b_{yz}D_{yz} + b_{zz}D_{zz}) 

The gradient field typically stays in the one gradient direction, in
this case the relationship between $b$, $\mathbf{g}$ and the $b_{ij}$ is as
follows. If we fill out the symmetric B-matrix as:
 
.. math::

   \mathbf{B} = \begin{pmatrix}
                 b_{xx} & b_{yx} & b_{yz}\\
                 b_{xy} & b_{yy} & b_{xz}\\
                 b_{xz} & b_{yz} & b_{zz}
                 \end{pmatrix}

then $\mathbf{B}$ is equal to the rank 2 tensor $\gamma^2 G^2 R
\mathbf{g} \mathbf{g}^T$. By performing an eigenvalue and
eigenvector decomposition of $\mathbf{B}$ we obtain

.. math::

   \mathbf{B} = \lambda_1\mathbf{v}_1\mathbf{v}_1^T +
                \lambda_2\mathbf{v}_2\mathbf{v}_2^T +
                \lambda_3\mathbf{v}_3\mathbf{v}_3^T, 

where only one of the $\lambda_i$, say $\lambda_1$, is (effectively)
non-zero. (Because the gradient is always a multiple of a constant
direction $\mathbf{B}$ is a effectively a rank 1 tensor.) Then
$\mathbf{g} = \pm\mathbf{v}_1$, and $b = \gamma^2 G^2 R =
\lambda_1$. The ``b-vector`` $\mathbf{b}$ is given by:

.. math::

   \mathbf{b}_{\mathrm{actual}} = \gamma^2 G^2 R \mathbf{g}_{\mathrm{actual}}
    = \lambda_1 \mathbf{v}_1.

Once we have $\mathbf{b}_{actual}$ we can calculate $b_{actual} =
\|\mathbf{b}_{actual}\|$ and $\mathbf{g}_{actual} = \mathbf{b}_{actual}
/ b_{actual}$. Various sofware packages (e.g. FSL's DFT-DTIFIT) expect
to get N x 3 and N x 1 arrays of $\mathbf{g}_{actual}$ (``bvecs``) and
$b_{actual}$ values (``bvals``) as their inputs.

=======================
... and what about 'q'?
=======================

Callaghan, Eccles and Xia (1988) showed that the signal from the
narrow pulse PGSE paradigm measured the Fourier transform of the
diffusion displacement propagator. Propagation space is measured in
displacement per unit time $(\mathrm{mm}.\mathrm{seconds}^{-1})$. They
named the reciprocal space ``q-space`` with units of
$\mathrm{seconds}.\mathrm{mm}^{-1}$. 

.. math::
   :label: fourier

   q = \gamma \delta G /{2\pi}

.. math::

   b = 4 \pi^2 q^2 \Delta

Diffusion spectroscopy measures signal over a wide range of $b$-values
(or $q$-values) and diffusion times ($\Delta$) and performs a $q$-space
analysis (Fourier transform of the diffusion signal decay).

There remains a bit of mystery as to how $\mathbf{q}$ (as a vector in
$q$-space) is specified for other paradigms. We think that (a) it only
matters up to a scale factor, and (b) we can loosely identify
$\mathbf{q}$ with $b\mathbf{g}$, where $\mathbf{g}$ is the unit
vector in the gradient direction.

