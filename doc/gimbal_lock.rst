.. _gimbal-lock:

=============
 Gimbal lock
=============

See also: http://en.wikipedia.org/wiki/Gimbal_lock

Euler angles have a major deficiency, and that is, that it is possible,
in some rotation sequences, to reach a situation where two of the three
Euler angles cause rotation around the same axis of the object.  In the
case below, rotation around the $x$ axis becomes indistinguishable in
its effect from rotation around the $z$ axis, so the $z$ and $x$ axis
angles collapse into one transformation, and the rotation reduces from
three degrees of freedom to two.

Imagine that we are using the Euler angle convention of starting with a
rotation around the $x$ axis, followed by the $y$ axis, followed by the
$z$ axis.

Here we see a Spitfire aircraft, flying across the screen.  The $x$ axis
is left to right (tail to nose), the $y$ axis is from the left wing tip
to the right wing tip (going away from the screen), and the $z$ axis is
from bottom to top:

.. image:: images/spitfire_0.png

Imagine we wanted to do a slight roll with the left wing tilting down
(rotation about $x$) like this:

.. image:: images/spitfire_x.png

followed by a violent pitch so we are pointing straight up (rotation
around $y$ axis):

.. image:: images/spitfire_y.png

Now we'd like to do a turn of the nose towards the viewer (and the tail
away from the viewer):

.. image:: images/spitfire_hoped.png

But, wait, let's go back over that again.  Look at the result of the
rotation around the $y$ axis.  Notice that the $x$ axis, as was, is now
aligned with the $z$ axis, as it is now.  Rotating around the $z$ axis
will have exactly the same effect as adding an extra rotation around the
$x$ axis at the beginning.  That means that when there is a $y$ axis
rotation that rotates the $x$ axis onto the $z$ axis (a rotation of
$\pm\pi/2$ around the $y$ axis) - the $x$ and $y$ axes are "locked"
together.

Mathematics of gimbal lock
==========================

We see gimbal lock for this type of Euler axis convention, when
$\cos(\beta) = 0$, where $\beta$ is the angle of rotation around the $y$
axis.  By "this type of convention" we mean using rotation around all 3
of the $x$, $y$ and $z$ axes, rather than using the same axis twice -
e.g. the physics convention of $z$ followed by $x$ followed by $z$ axis
rotation (the physics convention has different properties to its gimbal
lock).

We can show how gimbal lock works by creating a rotation matrix for the
three component rotations. Recall that, for a rotation of $\alpha$
radians around $x$, followed by a rotation $\beta$ around $y$, followed
by rotation $\gamma$ around $z$, the rotation matrix $R$ is:

.. math::

   R = \left(\begin{smallmatrix}\operatorname{cos}\left(\beta\right) \operatorname{cos}\left(\gamma\right) & - \operatorname{cos}\left(\alpha\right) \operatorname{sin}\left(\gamma\right) + \operatorname{cos}\left(\gamma\right) \operatorname{sin}\left(\alpha\right) \operatorname{sin}\left(\beta\right) & \operatorname{sin}\left(\alpha\right) \operatorname{sin}\left(\gamma\right) + \operatorname{cos}\left(\alpha\right) \operatorname{cos}\left(\gamma\right) \operatorname{sin}\left(\beta\right)\\\operatorname{cos}\left(\beta\right) \operatorname{sin}\left(\gamma\right) & \operatorname{cos}\left(\alpha\right) \operatorname{cos}\left(\gamma\right) + \operatorname{sin}\left(\alpha\right) \operatorname{sin}\left(\beta\right) \operatorname{sin}\left(\gamma\right) &- \operatorname{cos}\left(\gamma\right) \operatorname{sin}\left(\alpha\right) + \operatorname{cos}\left(\alpha\right) \operatorname{sin}\left(\beta\right) \operatorname{sin}\left(\gamma\right)\\- \operatorname{sin}\left(\beta\right) & \operatorname{cos}\left(\beta\right) \operatorname{sin}\left(\alpha\right) & \operatorname{cos}\left(\alpha\right) \operatorname{cos}\left(\beta\right)\end{smallmatrix}\right)

When $\cos(\beta) = 0$, $\sin(\beta) = \pm1$ and $R$ simplifies to:

.. math::

     R = \left(\begin{smallmatrix}0 & - \operatorname{cos}\left(\alpha\right) \operatorname{sin}\left(\gamma\right) + \pm{1} \operatorname{cos}\left(\gamma\right) \operatorname{sin}\left(\alpha\right) & \operatorname{sin}\left(\alpha\right) \operatorname{sin}\left(\gamma\right) + \pm{1} \operatorname{cos}\left(\alpha\right) \operatorname{cos}\left(\gamma\right)\\0 & \operatorname{cos}\left(\alpha\right) \operatorname{cos}\left(\gamma\right) + \pm{1} \operatorname{sin}\left(\alpha\right) \operatorname{sin}\left(\gamma\right) & - \operatorname{cos}\left(\gamma\right) \operatorname{sin}\left(\alpha\right) + \pm{1} \operatorname{cos}\left(\alpha\right) \operatorname{sin}\left(\gamma\right)\\- \pm{1} & 0 & 0\end{smallmatrix}\right)

When $\sin(\beta) = 1$:

.. math::

   R = \left(\begin{smallmatrix}0 & \operatorname{cos}\left(\gamma\right) \operatorname{sin}\left(\alpha\right) - \operatorname{cos}\left(\alpha\right) \operatorname{sin}\left(\gamma\right) & \operatorname{cos}\left(\alpha\right) \operatorname{cos}\left(\gamma\right) + \operatorname{sin}\left(\alpha\right) \operatorname{sin}\left(\gamma\right)\\0 & \operatorname{cos}\left(\alpha\right) \operatorname{cos}\left(\gamma\right) + \operatorname{sin}\left(\alpha\right) \operatorname{sin}\left(\gamma\right) & \operatorname{cos}\left(\alpha\right) \operatorname{sin}\left(\gamma\right) - \operatorname{cos}\left(\gamma\right) \operatorname{sin}\left(\alpha\right)\\-1 & 0 & 0\end{smallmatrix}\right)

From the `angle sum and difference identities
<http://en.wikipedia.org/wiki/List_of_trigonometric_identities#Angle_sum_and_difference_identities>`_
(see also `geometric proof
<http://www.themathpage.com/atrig/sum-proof.htm>`_, `Mathworld treatment
<http://mathworld.wolfram.com/TrigonometricAdditionFormulas.html>`_) we
remind ourselves that, for any two angles $\alpha$ and $\beta$:

.. math::

   \sin(\alpha \pm \beta) = \sin \alpha \cos \beta \pm \cos \alpha \sin \beta \,

   \cos(\alpha \pm \beta) = \cos \alpha \cos \beta \mp \sin \alpha \sin \beta

We can rewrite $R$ as:

.. math::

    R = \left(\begin{smallmatrix}0 & V_{1} & V_{2}\\0 & V_{2} & - V_{1}\\-1 & 0 & 0\end{smallmatrix}\right)

where:

.. math::

    V_1 = \operatorname{cos}\left(\gamma\right) \operatorname{sin}\left(\alpha\right) - \operatorname{cos}\left(\alpha\right) \operatorname{sin}\left(\gamma\right) = \sin(\alpha - \gamma) \,

    V_2 =  \operatorname{cos}\left(\alpha\right) \operatorname{cos}\left(\gamma\right) + \operatorname{sin}\left(\alpha\right) \operatorname{sin}\left(\gamma\right) = \cos(\alpha - \gamma)

We immediately see that $\alpha$ and $\gamma$ are going to lead the same
transformation - the mathematical expression of the observation on the
spitfire above, that rotation around the $x$ axis is equivalent to
rotation about the $z$ axis.

It's easy to do the same set of reductions, with the same conclusion,
for the case where $\sin(\beta) = -1$ - see
http://www.gregslabaugh.name/publications/euler.pdf.

