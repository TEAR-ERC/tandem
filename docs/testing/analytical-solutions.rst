Analytical solutions
====================

This page gives the exact solutions that the integration tests use as ground truth.
Each solution is also written in the Lua file of its test (``solution`` and
``solution_jacobian``), and ``app/static`` / ``app/tandem`` compute the error against it.
For the viscoelastic tests the closed form is in the pytest module.

Symbols: :math:`\mu` and :math:`\lambda` are the Lame parameters, :math:`\nu` is Poisson's ratio,
:math:`E = \mu(3\lambda + 2\mu)/(\lambda + \mu)` is Young's modulus.

Many of these solutions are linear (or affine) in space. The DG space contains every
linear field for every degree :math:`p \ge 1`, so the discrete solution must equal the exact one up to
the solver tolerance and round-off, on any mesh. A test with such a solution can therefore use the
absolute limit ``1e-8`` instead of a convergence rate.


Static elasticity
-----------------

.. _as-circular-hole:

Plate with a circular hole (2D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Used by: static regression, static convergence, parallel consistency (2D).
File: ``test/reference_configs/2D/circular_hole.lua``.

An infinite plane-strain plate with a hole of radius :math:`a = 0.2`, under far-field
pure shear :math:`\sigma_{xy} = S = 1`; :math:`\mu = 1`, :math:`\nu = 0.25`.
The domain is the annulus :math:`a \le r \le 1`. The hole (tag 1) is traction free and the
outer circle (tag 5) has the exact solution as Dirichlet data. In polar coordinates:

.. math::

   u_r = \frac{S}{2\mu r^3}\left(r^4 - 4a^2r^2(\nu - 1) - a^4\right)\sin 2\theta,

.. math::

   u_\theta = \frac{S}{2\mu r^3}\left(r^4 + 2a^2r^2 - 4a^2\nu r^2 + a^4\right)\cos 2\theta.

For :math:`r \to \infty` this tends to :math:`\boldsymbol{u} = \frac{S}{2\mu}(y, x)`,
the uniform shear field. The field is not polynomial, so the error converges with the mesh size.

.. _as-spherical-hole:

Body with a spherical hole (3D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Used by: static regression, static convergence, parallel consistency (3D).
File: ``test/reference_configs/3D/spherical_hole.lua``.

An infinite body with a spherical hole of radius :math:`a = 0.5`, under far-field uniaxial
tension :math:`\sigma_{zz} = S = 1`; :math:`\mu = 1`, :math:`\nu = 0.25`. With :math:`R` the radius and :math:`\beta` the polar angle
from the :math:`z` axis:

.. math::

   \begin{aligned}
   u_R &= \frac{S}{8\mu R^4 (5\nu - 7)(1 + \nu)}\Big[
          (1+\nu)\,a^3R^2(10\nu - 13) \\
       &\qquad + (1+\nu)\big(9a^5 + R^5(10\nu - 14) + a^3R^2(20\nu - 25)\big)\cos 2\beta \\
       &\qquad + 3(1+\nu)\,a^5 + 2(5\nu - 7)(1 - \nu)\,R^5 \Big], \\
   u_\beta &= -\frac{S}{4\mu R^4 (5\nu - 7)}\big(-3a^5 + (5\nu - 7)R^5 + a^3R^2(10\nu - 5)\big)\sin 2\beta.
   \end{aligned}

For :math:`R \to \infty`, :math:`u_R \to \frac{S R}{2E}\big((1+\nu)\cos 2\beta + 1 - \nu\big)`,
the uniaxial field :math:`\varepsilon_{zz} = S/E`, :math:`\varepsilon_{xx} = \varepsilon_{yy} = -\nu S/E`.

.. _as-free-slip:

Block with free slip sides
^^^^^^^^^^^^^^^^^^^^^^^^^^

Used by: free slip BC test. Files: ``examples/elasticity/{2d,3d}/free_slip.lua``.

The unit square :math:`[0, L]\times[0, H]` (:math:`L = H = 1`), :math:`\mu = 1`, :math:`\lambda = 2`.
The sides have free slip with :math:`u_n = g`, the bottom has free slip with :math:`u_n = 0`,
the top is traction free. With :math:`\varepsilon = 2g/L`:

.. math::

   \text{2D:}\quad u_x = \varepsilon\,(x - L/2), \qquad
   u_y = -\frac{\lambda}{\lambda + 2\mu}\,\varepsilon\, y.

.. math::

   \text{3D:}\quad u_x = \varepsilon\,(x - L/2), \qquad u_y = \varepsilon\,(y - L/2), \qquad
   u_z = -\frac{2\lambda}{\lambda + 2\mu}\,\varepsilon\, z.

The stress is uniform and has no shear component, so the sides and the bottom carry no tangential
traction. The vertical strain makes :math:`\sigma_{zz} = 0` (2D: :math:`\sigma_{yy} = 0`),
so the top is traction free. The field is linear.

.. _as-traction-patch:

Block in uniaxial tension (2D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Used by: traction patch BC test. File: ``examples/elasticity/2d/traction_patch.lua``.

The unit square, :math:`\mu = 1`, :math:`\lambda = 2`, traction :math:`(s, 0)` on the right.
The exact state is uniform uniaxial stress in plane strain,
:math:`\sigma_{xx} = s`, :math:`\sigma_{yy} = \sigma_{xy} = 0`:

.. math::

   u_x = \varepsilon_{xx}\,x, \qquad u_y = \varepsilon_{yy}\,(y - H/2), \qquad
   \varepsilon_{xx} = \frac{s\,(\lambda + 2\mu)}{4\mu(\lambda + \mu)}, \qquad
   \varepsilon_{yy} = -\frac{\lambda}{\lambda + 2\mu}\,\varepsilon_{xx}.

The left side has this field as Dirichlet data, so the lateral contraction is not blocked.
The field is linear.

.. _as-rod:

Rod under end traction (3D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Used by: traction BC test (3D). File: ``test/reference_configs/3D/cantilever_rod.lua``.

A rod of length :math:`L` under the end traction :math:`\sigma_{xx} = \sigma_0`. In uniaxial
stress the end displacement is

.. math::

   u_x(L) = \frac{\sigma_0 L}{E}
          = \frac{0.01\ \text{GPa}\cdot 10^{-4}\ \text{km}}{200.1\ \text{GPa}}
          \approx 5\times10^{-9}\ \text{km}
          \quad (\mu = 77,\ \lambda = 115\ \text{GPa}).

The test clamps all components at :math:`x = 0`. This blocks the lateral contraction there,
so the stress is uniaxial only away from the clamp and :math:`\sigma_0 L/E` is an approximation.


Poisson
-------

.. _as-volume-tagging:

Two materials in series
^^^^^^^^^^^^^^^^^^^^^^^

Used by: volume tagging correctness test.
File: ``test/reference_configs/2D/volume_tagging_correctness.lua``.

:math:`-\nabla\cdot(k\nabla\phi) = 0` on :math:`[-1, 1]\times[0, 1]`, with :math:`k_A = 1` for
:math:`x < 0` (tag 1) and :math:`k_B = 2` for :math:`x > 0`, :math:`\phi(-1) = 0`, :math:`\phi(1) = 1`.
Continuity of :math:`\phi` and of the flux :math:`k\,\partial_x\phi` at :math:`x = 0` gives

.. math::

   \phi^* = \frac{k_B\phi_1 + k_A\phi_0}{k_A + k_B} = \frac{2}{3}, \qquad
   \phi(x) = \begin{cases}
     \phi^*\,(x + 1), & x \le 0, \\
     \phi^* + (1 - \phi^*)\,x, & x > 0.
   \end{cases}

The field is linear on each side, and the mesh follows the interface, so it is exact.


Viscoelasticity
---------------

The viscoelastic material is a standard linear solid (SLS): a spring :math:`\mu_0` in parallel
with a Maxwell branch (spring :math:`\mu_1` in series with a dashpot :math:`\eta`).
The relaxation time is :math:`\tau = \eta/\mu_1`.
The internal strain :math:`q` of the Maxwell branch follows

.. math::

   \dot q + \frac{q}{\tau} = \dot\varepsilon^{dev}, \qquad
   \sigma = \sigma_{\text{elastic}}(\mu_0) + 2\mu_1 q \quad (\text{deviatoric part}).

What tandem does with this law in one time step :math:`\Delta t` (needed to understand the measured errors):

.. math::

   g = \frac{1 - e^{-\Delta t/\tau}}{\Delta t/\tau}, \qquad
   q^{n+1} = e^{-\Delta t/\tau} q^n + g\,(\varepsilon^{dev,n+1} - \varepsilon^{dev,n}),

and the step solves an elastic problem with the effective moduli

.. math::

   A(\Delta t) = \lambda + \tfrac{2}{3}\mu_1(1 - g), \qquad B(\Delta t) = \mu_0 + \mu_1 g.

The update is exact when the strain is linear in time within the step, and second order otherwise.
At :math:`t = 0` the dashpot cannot move, so tandem solves the first step with the unrelaxed
modulus :math:`\mu_0 + \mu_1`.

.. _as-simple-shear:

Simple shear
^^^^^^^^^^^^

Used by: viscoelastic simple shear test. Closed form: ``SimpleShearSLS`` in
``test/test_viscoelastic_simple_shear.py``.

The shear strain is imposed: :math:`\gamma(t) = (2U_{max}/H)\,s(t)`. The stress is

.. math::

   \sigma_{xy}(t) = \mu_0\gamma(t) + \mu_1 q(t), \qquad
   q(t) = \gamma(0)\,e^{-t/\tau} + \int_0^t e^{-(t-s)/\tau}\,\gamma'(s)\,ds.

With :math:`\gamma_0 = 2U_{max}/H` and :math:`\beta = 1/\tau`, the three loadings give

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - case
     - :math:`\gamma(t)`
     - :math:`q(t)` before a drop
   * - constant strain
     - :math:`\gamma_0`
     - :math:`\gamma_0\,e^{-t/\tau}`
   * - constant strain rate
     - :math:`\gamma_0\,t`
     - :math:`\gamma_0\,\tau\,(1 - e^{-t/\tau})`
   * - sinusoid
     - :math:`\gamma_0\sin\omega t`
     - :math:`\dfrac{\gamma_0\,\omega}{\beta^2 + \omega^2}\big(\beta\cos\omega t + \omega\sin\omega t - \beta e^{-\beta t}\big)`

A drop of the strain to zero at :math:`T_{hold}` subtracts the strain just before the drop
from :math:`q`, and after it only the memory relaxes:

.. math::

   \sigma_{xy}(t) = \mu_1\big(q(T_{hold}) - \gamma(T_{hold})\big)\,e^{-(t - T_{hold})/\tau},
   \qquad t > T_{hold}.

The drop is at :math:`t \le T_{hold}`, the same as in ``simple_shear.lua``. With :math:`<` the test would
compare a loaded sample against the unloaded branch.

The constant-strain and constant-strain-rate histories (without drop) are linear in time
within each step, so tandem must reproduce them exactly (to ``1e-10``).

.. _as-mms:

Manufactured solution
^^^^^^^^^^^^^^^^^^^^^

Used by: viscoelastic MMS test. File: ``examples/tandem/viscoelasticity/2d/mms.lua``.

:math:`\mu_0 = 2`, :math:`\mu_1 = 1.5`, :math:`\tau = 0.5`, :math:`a = 1`. The displacement starts
from rest and is divergence free (a stream function :math:`\psi`):

.. math::

   \boldsymbol{u}(\boldsymbol{x}, t) = (1 - e^{-at})\,\boldsymbol{U}(\boldsymbol{x}), \qquad
   \boldsymbol{U} = (\partial_y\psi, -\partial_x\psi), \qquad \nabla\cdot\boldsymbol{U} = 0.

Then :math:`\operatorname{tr}\varepsilon = 0`, so only the deviatoric memory path acts.
With :math:`q(0) = 0` the memory law gives
:math:`q = A_q\,E\,(e^{-at} - e^{-t/\tau})`, :math:`A_q = a\tau/(1 - a\tau)` (here :math:`A_q = 1`),
and the stress is :math:`\sigma = M(t)\,E(\boldsymbol{x})` with :math:`E` the strain of :math:`\boldsymbol{U}` and

.. math::

   M(t) = 2\mu_0(1 - e^{-at}) + 2\mu_1 A_q\,(e^{-at} - e^{-t/\tau})
        = 4(1 - e^{-t}) + 3(e^{-t} - e^{-2t}).

The body force is :math:`\boldsymbol{f} = -\nabla\cdot\sigma`. The two scenarios:

``mms`` (spatial study)
   :math:`\psi = \sin\pi x\sin\pi y`:
   :math:`\boldsymbol{U} = (\pi\sin\pi x\cos\pi y,\ -\pi\cos\pi x\sin\pi y)`,
   :math:`\boldsymbol{f} = M(t)\,\pi^3(\sin\pi x\cos\pi y,\ -\cos\pi x\sin\pi y)`.
   Not in the discrete space, so the L2 error shows the spatial order :math:`p + 1`.

``mms_poly`` (temporal study)
   :math:`\psi = x^2y + xy^2`:
   :math:`\boldsymbol{U} = (x^2 + 2xy,\ -(2xy + y^2))`, :math:`\boldsymbol{f} = M(t)\,(-1, 1)`.
   Quadratic, so for :math:`p \ge 2` the spatial error is zero and the L2 error is the time error alone
   (order 2).

The solution must start from rest. The code starts :math:`\varepsilon^{dev,n}` at zero, so a field that is
non-zero at :math:`t = 0` is seen as a jump, and the first step puts :math:`q = g\,\varepsilon(0)` into the memory.
A manufactured solution with :math:`q(0) = 0` then disagrees with the code at every later time.

.. _as-zener:

Rod creep and recovery (Zener solid)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Used by: viscoelastic rod test. Closed form: ``ZenerRod`` in ``test/test_viscoelastic_cantilever.py``.

Under uniaxial stress the SLS in shear with bulk modulus :math:`K` behaves as a Zener solid in
tension, with :math:`G_0 = \mu_0 + \mu_1` and

.. math::

   E_0 = \frac{9KG_0}{3K + G_0}, \qquad
   E_\infty = \frac{9K\mu_0}{3K + \mu_0}, \qquad
   \tau_c = \tau\,\frac{G_0}{\mu_0}.

The creep compliance is

.. math::

   D(t) = \frac{1}{E_\infty} - \frac{E_0 - E_\infty}{E_0E_\infty}\,e^{-t/\tau_c},

and for the load :math:`s_0` on :math:`0 \le t \le t_1`:

.. math::

   \varepsilon_{xx}(t) = \begin{cases}
     s_0\,D(t), & t \le t_1, \\
     s_0\,\big(D(t) - D(t - t_1)\big), & t > t_1.
   \end{cases}

With the test values, :math:`\tau_c = 1010` s and the strain grows about ten times during creep.

**Expected overshoot at unloading.** The step that removes the load sees the effective modulus
:math:`B(\Delta t) = \mu_0 + \mu_1 g` instead of :math:`G_0`, so the elastic rebound is too large by

.. math::

   \Delta\varepsilon = s_0\left(\frac{1}{E_{eff}} - \frac{1}{E_0}\right), \qquad
   E_{eff} = \frac{9KB}{3K + B}, \qquad \Delta\varepsilon = 4.5064\times10^{-4}\ \ (\theta = 0.1).

This is a first-order error of the time scheme at a load jump. The test checks that the
measured error is this prediction (measured: ``1.00023`` x the prediction).
