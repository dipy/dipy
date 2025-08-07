==========
 Glossary
==========

.. glossary::

   Affine matrix
      A matrix implementing an :term:`affine transformation` in
      :term:`homogeneous coordinates`.  For a 3 dimensional transform, the
      matrix is shape 4 by 4.

   Affine transformation
      See `wikipedia affine`_ definition.  An affine transformation is a
      :term:`linear transformation` followed by a translation.

   Axis angle
      A representation of rotation.  See: `wikipedia axis angle`_ .
      From Euler's rotation theorem we know that any rotation or
      sequence of rotations can be represented by a single rotation
      about an axis.  The axis $\boldsymbol{\hat{u}}$ is a :term:`unit
      vector`.  The angle is $\theta$.  The :term:`rotation vector` is a
      more compact representation of $\theta$ and
      $\boldsymbol{\hat{u}}$.

   Euclidean norm
      Also called Euclidean length, or L2 norm.  The Euclidean norm
      $\|\mathbf{x}\|$ of a vector $\mathbf{x}$ is given by:

      .. math::

         \|\mathbf{x}\| := \sqrt{x_1^2 + \cdots + x_n^2}

      Pure Pythagoras.

   Euler angles
      See: `wikipedia Euler angles`_ and `Mathworld Euler angles`_.

   Gimbal lock
      See :ref:`gimbal-lock`

   Homogeneous coordinates
      See `wikipedia homogeneous coordinates`_

   Linear transformation
      A linear transformation is one that preserves lines - that is, if
      any three points are on a line before transformation, they are
      also on a line after transformation.  See `wikipedia linear
      transform`_.  Rotation, scaling and shear are linear
      transformations.

   Quaternion
      See: `wikipedia quaternion`_.  An extension of the complex numbers
      that can represent a rotation.  Quaternions have 4 values, $w, x,
      y, z$.  $w$ is the *real* part of the quaternion and the vector
      $x, y, z$ is the *vector* part of the quaternion.  Quaternions are
      less intuitive to visualize than :term:`Euler angles` but do not
      suffer from :term:`gimbal lock` and are often used for rapid
      interpolation of rotations.

   Reflection
      A transformation that can be thought of as transforming an object
      to its mirror image.  The mirror in the transformation is a plane.
      A plan can be defined with a point and a vector normal to the
      plane.  See `wikipedia reflection`_.

   Rotation matrix
      See `wikipedia rotation matrix`_.  A rotation matrix is a matrix
      implementing a rotation.  Rotation matrices are square and
      orthogonal.  That means, that the rotation matrix $R$ has columns
      and rows that are :term:`unit vector`, and where $R^T R = I$ ($R^T$ is
      the transpose and $I$ is the identity matrix).  Therefore $R^T =
      R^{-1}$ ($R^{-1}$ is the inverse).  Rotation matrices also have a
      determinant of $1$.

   Rotation vector
      A representation of an :term:`axis angle` rotation. The angle
      $\theta$ and unit vector axis $\boldsymbol{\hat{u}}$ are stored in a
      *rotation vector* $\boldsymbol{u}$, such that:

      .. math::

         \theta =  \|\boldsymbol{u}\| \,

         \boldsymbol{\hat{u}} = \frac{\boldsymbol{u}}{\|\boldsymbol{u}\|}

      where $\|\boldsymbol{u}\|$ is the :term:`Euclidean norm` of
      $\boldsymbol{u}$

   Shear matrix
      Square matrix that results in shearing transforms - see
      `wikipedia shear matrix`_.

   Unit vector
      A vector $\boldsymbol{\hat{u}}$ with a :term:`Euclidean norm`
      of 1.  Normalized vector is a synonym.  The "hat" over the
      $\boldsymbol{\hat{u}}$ is a convention to express the fact that it
      is a unit vector.

.. include:: links_names.inc
