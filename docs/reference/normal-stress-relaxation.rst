Normal stress relaxation and bounds
===================================

In a standard Tandem model the transient normal stress change :math:`\sigma_n` computed by the
elastic solve acts on the fault for all times. In geometries with a lot of normal stress
variation, e.g. at fault step-overs, this change can accumulate over many earthquake cycles
and push the absolute normal stress to unrealistically high or low values.
Tandem provides two optional mechanisms to deal with this.
Both are off unless they are set in the Lua script, in which case the behaviour is identical
to a model without them.

Maxwell relaxation (``t_relax``)
--------------------------------

The friction law uses an effective transient normal stress

.. math::
   \sigma_n^{\mathrm{eff}} = \sigma_n + q,

where :math:`\sigma_n` is the elastic normal stress change (including ``delta_sn``, if given) and
:math:`q` is an additional internal variable per fault node with :math:`q(0) = 0` that evolves as

.. math::
   \dot{q} = -\frac{\sigma_n + q}{T_r}.

This is equivalent to the Maxwell relaxation law

.. math::
   \dot{\sigma}_n^{\mathrm{eff}} = \dot{\sigma}_n - \frac{\sigma_n^{\mathrm{eff}}}{T_r},

with the relaxation time :math:`T_r` (``t_relax``), but only requires :math:`\sigma_n` and not
its time derivative. A sudden elastic change of :math:`\sigma_n` first acts with its full size
and then decays exponentially with e-folding time :math:`T_r`. In general,
:math:`\sigma_n^{\mathrm{eff}}` is the sum of all past changes of :math:`\sigma_n`, each decayed by
:math:`e^{-(t - t_i)/T_r}`.

The limits are

* :math:`T_r \to \infty`: :math:`q = 0`, i.e. no relaxation (purely elastic),
* :math:`T_r \to 0`: :math:`q = -\sigma_n`, i.e. :math:`\sigma_n^{\mathrm{eff}} = 0` and the
  transient normal stress is removed completely.

Choosing the relaxation time
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:math:`T_r` should be much longer than the time scale of the coseismic and early postseismic
phase (seconds to days), since otherwise the normal stress changes that govern rupture
propagation are removed as well. It should be much shorter than the recurrence interval
(decades to centuries), otherwise nothing is relaxed. In between, values of months to a few
decades are reasonable. For a viscoelastic interpretation :math:`T_r = \eta / \mu`, with
viscosity :math:`\eta` and shear modulus :math:`\mu`.
It is recommended to test the sensitivity of the results to :math:`T_r`.

A very small :math:`T_r` compared to the time step makes the equation for :math:`q` stiff.

Usage
^^^^^

``t_relax`` is a functional of space, given in the time unit of the model (seconds).
It can vary along the fault:

.. code:: lua

   -- relaxation time of about 1 year
   function MyScenario:t_relax(x, y)
       return 3.15e7
   end

Defining ``t_relax`` adds one value per fault node to the state vector (see below).
Omitting it, or returning ``math.huge``, disables the relaxation.
``t_relax`` must be positive.

Bounds on the normal stress (``sn_min``, ``sn_max``)
----------------------------------------------------

The absolute effective normal stress (positive in compression) used by the friction law is

.. math::
   \hat{\sigma}_n = \min\left(\max\left(\sigma_n^0 - \sigma_n^{\mathrm{eff}},
                    \sigma_n^{\min}\right), \sigma_n^{\max}\right).

A lower bound :math:`\sigma_n^{\min} > 0` avoids (near-)zero normal stress, for which the
rate-and-state friction becomes degenerate and the slip-rate solve may fail to converge, and the
tension branch (:math:`\hat{\sigma}_n \le 0`, fault opening) is never reached.
An upper bound avoids excessive compression. Both are constants:

.. code:: lua

   MyScenario.sn_min = 1.0   -- MPa
   MyScenario.sn_max = 200.0 -- MPa

If they are omitted there are no bounds. ``sn_min`` must not exceed ``sn_max``.

Notes
-----

* The bounds are applied to the effective normal stress seen by the friction law (after the
  relaxation). The relaxation equation for :math:`q` is not modified, so :math:`q` cannot wind
  up at a bound. The elastic traction is not modified either, hence there is a small inconsistency
  between the elastic solution and the friction law while a bound is active.
* The clamp is continuous but not smooth. If the time stepping struggles when a bound is
  switching on and off, consider a smoother bound.
* With the lower bound active the fault can no longer open.
* The ``normal-stress`` output is the (bounded) effective normal stress
  :math:`\hat{\sigma}_n`.
* Per fault node the state vector contains the slip components, the state variable
  :math:`\psi`, and, only if ``t_relax`` is defined, :math:`q` (stored last).
  Checkpoints of models with and without ``t_relax`` are therefore not interchangeable.
  The additional variable is also subject to the error control of the time integrator, and it is
  in units of stress.
* The parameter ``relax_rate`` :math:`= 1/T_r` is added at the end of the fault parameter output.
