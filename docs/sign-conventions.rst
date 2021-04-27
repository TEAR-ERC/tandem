Sign conventions
================

Slip is defined as

.. math::
   \boldsymbol{S} = \boldsymbol{u}^- - \boldsymbol{u}^+

Let the orthogonal basis :math:`\boldsymbol{n},\boldsymbol{d},\boldsymbol{s}` be given,
where normal :math:`\boldsymbol{n}` points from the "-"-side to the "+"-side,
:math:`\boldsymbol{d}` is the dip direction, and :math:`\boldsymbol{s}` is the strike direction.
The slip-rate vector is defined as

.. math::
   \boldsymbol{V} = [\boldsymbol{\dot{S}}\cdot \boldsymbol{d}, \boldsymbol{\dot{S}}\cdot \boldsymbol{s}],

the shear traction vector is

.. math::
   \boldsymbol{\tau} = [\boldsymbol{d}\cdot \sigma\boldsymbol{n},
                        \boldsymbol{s}\cdot \sigma\boldsymbol{n}],

and the normal stress is given by

.. math::
   \sigma_n = \boldsymbol{n}\cdot \sigma \boldsymbol{n}.

Note that in 2D we drop the second component of the slip-rate and shear traction vector.

The friction law is given by

.. math::
   -(\boldsymbol{\tau}^0 + \boldsymbol{\tau}) =
      (\sigma_n^0 - \sigma_n) f(|\boldsymbol{V}|,\psi)\frac{\boldsymbol{V}}{|\boldsymbol{V}|} +
                        \eta \boldsymbol{V},

where :math:`\boldsymbol{\tau}^0` and :math:`\sigma_n^0` are pre-stresses.
We take :math:`\sigma_n^0` to be positive in compression, thus the sign is different to
:math:`\sigma_n`.
