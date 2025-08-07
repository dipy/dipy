================================
 The B matrix and Siemens DICOM
================================

This is a short note to explain the nature of the ``B_matrix`` found in the
Siemens private (CSA) fields of the DICOM headers of a diffusion-weighted
acquisition.  We try to explain the relationship between the ``B_matrix`` and
the *b value* and the *gradient vector*.  The acquisition is made with a planned
(requested) b value - say $b_{req} = 1000$, and with a requested gradient
direction $\mathbf{g}_{req} = [g_x, g_y, g_z]$ (supposedly a unit vector).

Note that here we're using $\mathbf{q}$ in the sense of an approximation
to a vector in $q$ space.  Other people use $\mathbf{b}$ for the same
concept, but we've chosen $\mathbf{q}$ to make the exposition clearer.

For some purposes, we want the q vector $\mathbf{q}_{actual}$ which is
equal to $b_{actual} . \mathbf{g}_{actual}$. We need to be aware that
$b_{actual}$ and $\mathbf{g}_{actual}$ may be different from the
$b_{req}$ and $\mathbf{g}_{req}$!  Though the Stejskal and Tanner
formula is available for the classic PGSE sequence, a different sequence
may be used (e.g. TRSE on Siemens Trio), and anyway, the ramps up and
down on the gradient field will not be rectangular. The Siemens scanner
software calculates the effective directional diffusion weighting of the
acquisition based on the temporal profile of the applied gradient
vector field. These are in the form of the 6 ``B_matrix`` values
$[b_{xx}, b_{xy}, b_{xz}, b_{yy}, b_{yz}, b_{zz}]$.

In this form they are suitable for use in a least squares estimation of
the diffusion tensor via the equations across the set of acquisitions:

.. math::

   \log(A(\mathbf{q})/A(0)) = -(b_{xx}D_{xx} + 2b_{xy}D_{xy} + 2b_{xz}D_{xz} + \
      b_{yy}D_{yy} + 2b_{yz}D_{yz} + b_{zz}D_{zz})

The gradient field typically stays in the one gradient direction, in
this case the relationship between $\mathbf{q}$ and the $b_{ij}$ is as
follows. If we fill out the symmetric B-matrix as:

.. math::

   \mathbf{B} = \begin{pmatrix}
                 b_{xx} & b_{yx} & b_{yz}\\
                 b_{xy} & b_{yy} & b_{xz}\\
                 b_{xz} & b_{yz} & b_{zz}
                 \end{pmatrix}

then $\mathbf{B}$ is equal to the rank 1 tensor
$b\mathbf{g}\mathbf{g}^T$. One of the ways to recover $b$ and $\mathbf{g}$,
and hence $\mathbf{q}$, from
$\mathbf{B}$ is to do a singular value decomposition of $\mathbf{B}:
\mathbf{B} = \lambda_1\mathbf{v}_1\mathbf{v}_1^T +
\lambda_2\mathbf{v}_2\mathbf{v}_2^T +
\lambda_3\mathbf{v}_3\mathbf{v}_3^T$, where only one of the $\lambda_i$,
say $\lambda_1$, is effectively non-zero. Then $b = \lambda_1$, $\mathbf{g} =
\pm\mathbf{v}_1,$ and $\mathbf{q} =
\pm\lambda_1\mathbf{v}_1.$ The choice of sign is arbitrary
(essentially we have a choice between two possible square roots of the
rank 1 tensor $\mathbf{B}$). Once we have $\mathbf{q}_{actual}$ we can
calculate $b_{actual} = |\mathbf{q}_{actual}|$ and $\mathbf{g}_{actual}
= \mathbf{q}_{actual} / b_{actual}$. Various software packages
(e.g. FSL's DFT-DTIFIT) expect to get 3 × N and 1 × N arrays of
$\mathbf{g}_{actual}$ and $b_{actual}$ values as their inputs.
