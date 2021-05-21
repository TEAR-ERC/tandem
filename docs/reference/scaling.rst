Equation scaling
================

When working with SI units in SEAS models numbers might get very large.
Rescaling the equations might be advantageous to avoid large round-off errors in finite precision.
In this section, we show how to properly scale the elasticity equations.

The linear elasticity equations in first order form are given by

.. math::

   \begin{aligned}
      \sigma_{ij} &= \lambda \delta_{ij}\frac{\partial u_k}{\partial x_k} +
         \mu\left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right) \\
      - \frac{\partial \sigma_{ij}}{\partial x_j} &= f_i
   \end{aligned}

We define scaled quantities

.. math::

   \bar{x}_i = \frac{x_i}{L}, \quad \bar{u}_i = \frac{u_i}{u_c}, \quad
       \bar{\sigma}_{ij} = \frac{\sigma_{ij}}{\sigma_c},

where :math:`L, u_c, \sigma_c` are :ref:`scaling constants<Scaling Example>`.
Inserting these into the linear elasticity equations gives

.. math::

   \begin{aligned}
      \sigma_c\bar{\sigma}_{ij} &= L^{-1}u_c\lambda
         \delta_{ij}\frac{\partial \bar{u}_k}{\partial \bar{x}_k} +
         L^{-1}u_c\mu\left(\frac{\partial \bar{u}_i}{\partial \bar{x}_j} +
                           \frac{\partial \bar{u}_j}{\partial \bar{x}_i}\right) \\
      - L^{-1}\sigma_c\frac{\partial \bar{\sigma}_{ij}}{\partial \bar{x}_j} &= f_i
   \end{aligned}

Multiplying the first equation with :math:`\sigma_c^{-1}`,
multiplying the second equation with :math:`L\sigma_c^{-1}`, and defining

.. math::

   \bar{\lambda} = \sigma_c^{-1}u_cL^{-1}\lambda, \quad
   \bar{\mu} = \sigma_c^{-1}u_cL^{-1}\mu, \quad
   \bar{f}_i = L\sigma_c^{-1} f_i

leads to

.. math::

   \begin{aligned}
      \bar{\sigma}_{ij} &= \bar{\lambda}
         \delta_{ij}\frac{\partial \bar{u}_k}{\partial \bar{x}_k} +
         \bar{\mu}\left(\frac{\partial \bar{u}_i}{\partial \bar{x}_j} +
                           \frac{\partial \bar{u}_j}{\partial \bar{x}_i}\right) \\
      - \frac{\partial \bar{\sigma}_{ij}}{\partial \bar{x}_j} &= \bar{f}_i
   \end{aligned}

That is, we recovered the original equations and we only need to scale
the mesh and the parameters.

.. _Scaling Example:

Example
-------
We change units with the scaling constants

.. math:: L = 10^3, \quad u_c = 1, \quad \sigma_c = 10^6

In the rescaled equations, the spatial dimension of the mesh is [km],
velocities are in [m/s], and stresses are in [MPa].
Parameters and source terms are scaled with

.. math::

   \bar{\lambda} = 10^{-9}\lambda, \quad
   \bar{\mu} = 10^{-9}\mu, \quad
   \bar{f}_i = 10^{-3} f_i

i.e. the Lamé parameters are given in [GPa] and force in [10\ :sup:`-3` N/m\ :sup:`-3`].
