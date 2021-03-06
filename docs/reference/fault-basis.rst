Fault basis
===========

Slip and slip-rate are defined with respect to a local fault basis.
In this document the conventions for the fault basis are introduced.
The direction of movement is defined in terms of the hanging wall and the
foot wall:

"The foot wall (hanging wall) is defined as the block below (above) the
fault plane. (...) the hanging wall moves up with respect to the foot
wall and the fault is known as **reverse**. (...) the opposite happens and
the fault is said to be **normal**." [J. Pujol, Elastic Wave Propagation
and Generation in Seismology]

The sign of the fault normal is chosen such that

.. math::

   n \cdot n_{\text{ref}} > 0.

We define that the fault normal points from the foot wall to the hanging wall.
In this way the reference normal :math:`n_{\text{ref}}` selects the foot and
the hanging wall.

The first component of the slip or slip-rate vector is defined w.r.t. to the
normal direction of the fault. Due to the no-opening condition the first
component is zero.

The third component of the slip or slip-rate vector is defined w.r.t.
to the strike direction. The latter is defined such that a hypothetical
observer standing on the fault looking in strike direction sees the hanging
wall on his right. Thus, the strike direction is

.. math::

   s := u \times n,

where :math:`u` is the direction of "up", given in the configuration file.
E.g. using the **enu** convention, up would be the vector :math:`u=(0, 0, 1)`.

The second component of the slip or slip-rate vector is defined w.r.t.
to the dip direction, which we define to point "down". That is, the
dip direction is

.. math::

   d := s \times n

Left-lateral, right-lateral, normal, reverse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The slip vector is given by :math:`u=[u_n] n + [u_d] d + [u_s] s`, where
the square bracket operator for a scalar field :math:`q` is defined as

.. math::

   [q] := q^- - q^+ = \lim_{\epsilon \rightarrow 0} q(x-\epsilon n) - q(x+\epsilon n)


Recall that the normal points from the foot wall to the hanging wall.
Thus, if :math:`[u_d] > 0` we have a **reverse** fault. Conversely,
if :math:`[u_d] < 0` we have a **normal** fault.

For strike slip fault, i.e. :math:`[u_s] \neq 0`, we have to distinguish
two cases:

"In a left-lateral (right-lateral) fault, an observer on one of the
walls will see the other wall moving to the left (right)." [J. Pujol,
Elastic Wave Propagation and Generation in Seismology]

If :math:`[u_s] > 0` then we have a right-lateral fault and if :math:`[u_s] < 0`
then we have a left-lateral fault.

.. _Vertical Fault:

Special-case: Flat fault
~~~~~~~~~~~~~~~~~~~~~~~~

Don't do that.
