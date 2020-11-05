Fault basis
===========

Slip and slip-rate are defined with respect to a local fault basis.
In this document the conventions for the fault basis are introduced.

Standard fault
~~~~~~~~~~~~~~

A "standard fault" has dip angle :math:`\delta \in (0,\pi/2)`, that is,
a standard fault is non-:ref:`vertical fault<Vertical Fault>` and
non-:ref:`flat fault<Flat Fault>`.
The direction of movement is defined in terms of the hanging wall and the
foot wall:

"The foot wall (hanging wall) is defined as the block below (above) the
fault plane. (...) the hanging wall moves up with respect to the foot
wall and the fault is known as **reverse**. (...) the opposite happens and
the fault is said to be **normal**." [J. Pujol, Elastic Wave Propagation
and Generation in Seismology]

Normal-, strike-, dip-direction
-------------------------------

In order to identify the hanging wall and the foot wall we need to know
the direction of "up". Let's call the up vector :math:`u` and assume it
was given in the configuration file.
E.g. using the **enu** convention, up would be the vector :math:`u=(0, 0, 1)`.
Tandem ensures that the normal :math:`n` points
from the foot wall to the hanging wall, i.e. the normal satisfies

.. math::

   n \cdot u > 0

The first component of the slip or slip-rate vector is defined w.r.t. to the
normal direction of the fault. Due to the no-opening condition the first
component is zero.

The third component of the slip or slip-rate vector is defined w.r.t.
to the strike direction. The latter is defined such that a hypothetical
observer standing on the fault looking in strike direction sees the hanging
wall on his right. Thus, the strike direction is

.. math::

   s := u \times n

The second component of the slip or slip-rate vector is defined w.r.t.
to the dip direction, which we define to point "down". That is, the
dip direction is

.. math::

   d := s \times n

Left-lateral, right-lateral, normal, reverse
--------------------------------------------

The slip vector is given by :math:`u=[u_n] n + [u_d] d + [u_s] s`, where
the square bracket operator for a scalar field :math:`q` is defined as

.. math::

   [q] := q^+ - q^- = \lim_{\epsilon \rightarrow 0} q(x+\epsilon n) - q(x-\epsilon n)


Recall that the normal points from the foot wall to the hanging wall.
Thus, if :math:`[u_d] > 0` then we have a **normal** fault. Conversely,
if :math:`[u_d] < 0` then we have a **reverse** fault.

For strike slip fault, i.e. :math:`[u_s] \neq 0`, we have to distinguish
two cases:

"In a left-lateral (right-lateral) fault, an observer on one of the
walls will see the other wall moving to the left (right)." [J. Pujol,
Elastic Wave Propagation and Generation in Seismology]

If :math:`[u_s] > 0` then we have a left-lateral fault and if :math:`[u_s] < 0`
then we have a right-lateral fault.

.. _Vertical Fault:

Special-case: Vertical fault
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a vertical fault (:math:`\delta = \pi/2`) hanging wall and foot wall do not
make sense anymore. Nevertheless, we can just assign a hanging wall (and consequently
a foot wall) such that we can keep above definitions of strike and dip direction.
In order to do so we introduce a reference normal :math:`n_{\text{ref}}`.
The normal :math:`n` is chosen such that

.. math::
   n\cdot n_{\text{ref}} \geq 0

.. _Flat Fault:

Special-case: Flat fault
~~~~~~~~~~~~~~~~~~~~~~~~

Just don't.
