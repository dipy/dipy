.. _spherical:

=======================
 Spherical coordinates
=======================

There are good dicussions of spherical coordinates in `wikipedia
spherical coordinate system`_ and `Mathworld spherical coordinate
system`_.

There is more information in the docstring for the
:func:`~dipy.core.geometry.sphere2cart` function.

Terms
=====

Origin
   Origin of the sphere

P
   The point represented by spherical coordinates

OP
   The line connecting the origin and P

radial distance
   or radius.  The Euclidean length of OP.

z axis
   The vertical of the sphere.  If we consider the sphere as a globe,
   then the z axis runs from south to north.  This is the zenith direction of the sphere. 

Reference plane
   The plane containing the origin and orthogonal to the z axis
   (zenith direction)

y axis
   Horizontal axis of the sphere, orthogonal to the z axis, on the
   reference plane.  West to east
   for a globe.

x axis
   Axis orthogonal to y and z axis, on the reference plane. For a globe,
   this will be a line
   from behind the globe through the origin towards us, the viewer.

Inclination angle
   The angle between OP and the z axis. This can also be called the
   polar angle, or the co-latitude.

Azimuth angle
   or azimuthal angle or longitude.  The angle between the projection of OP onto the
   reference plane and the x axis


The physics convention
======================

The radius is $r$, the inclination angle is $\theta$ and the azimuth angle is
$\phi$.  Spherical coordinates are specified by the tuple of $(r, \theta, \phi)$
in that order.

Here is a good illustration we made from the scripts kindly provided by `Jorge
Stolfi`_ on wikipedia.

.. _`Jorge Stolfi`: http://commons.wikimedia.org/wiki/User:Jorge_Stolfi

.. image:: spherical_coordinates.png

The formulae relating Cartesian coordinates $(x, y, z)$ to $r, \theta, \phi$ are:

.. math::

    r=\sqrt{x^2+y^2+z^2}

    \theta=\arccos\frac{z}{\sqrt{x^2+y^2+z^2}} 

    \phi = \operatorname{atan2}(y,x) 

and from $(r, \theta, \phi)$ to $(x, y, z)$:

.. math::

    x=r \, \sin\theta \, \cos\phi

    y=r \, \sin\theta \, \sin\phi

    z=r \, \cos\theta


The mathematics convention
==========================

See `wikipedia spherical coordinate system`_ .  The mathematics convention
reverses the meaning of $\theta$ and $\phi$ so that $\theta$ refers to the
azimuthal angle and $\phi$ refers to the inclination angle.

Matlab convention
=================

Matlab has functions ``sph2cart`` and ``cart2sph``.  These use the terms
``theta`` and ``phi``, but with a different meaning again from the standard
physics and mathematics conventions.   Here ``theta`` is the azimuth angle, as
for the mathematics convention, but ``phi`` is the angle between the reference
plane and OP.  This implies different formulae for the conversions between
Cartesian and spherical coordinates that are easy to derive.


.. include:: ../links_names.inc
