Lua scripting
=============

Material and frictional parameters used in Tandem simulation are written in Lua script. Here's key parameters and their descriptions:

+--------------------+---------------------------+-----------------------------------------------------------------+---------+---------+
| Name               | Format                    | Description                                                     | Unit    | Default |
+====================+===========================+=================================================================+=========+=========+
| **boundary**       | Functional (space, time)  | Boundary displacement condition governing Dirichlet boundary    | m       |         |
+--------------------+---------------------------+-----------------------------------------------------------------+---------+---------+
| **a**              | Functional (space)        | Rate-and-state direct effect parameter, a                       |         |         |
+--------------------+---------------------------+-----------------------------------------------------------------+---------+---------+
| **b**              | Constant                  | Rate-and-state evolution effect parameter, b                    |         |         |
+--------------------+---------------------------+-----------------------------------------------------------------+---------+---------+
| **L**              | Functional (space)        | Critical distance or state evolution distance (D_RS)            | m       |         |
+--------------------+---------------------------+-----------------------------------------------------------------+---------+---------+
| **V0**             | Constant                  | Reference slip rate                                             | m/s     |         |
+--------------------+---------------------------+-----------------------------------------------------------------+---------+---------+
| **f0**             | Constant                  | Reference friction coefficient                                  |         |         |
+--------------------+---------------------------+-----------------------------------------------------------------+---------+---------+
| **Vinit**          | Functional (space)        | Initial slip rate                                               | m/s     |         |
+--------------------+---------------------------+-----------------------------------------------------------------+---------+---------+
| **Sinit**          | Functional (space)        | Initial slip                                                    | m       |         |
+--------------------+---------------------------+-----------------------------------------------------------------+---------+---------+
| **mu**             | Functional (space)        | Shear modulus                                                   | GPa     | 1.0     |
+--------------------+---------------------------+-----------------------------------------------------------------+---------+---------+
| **lam**            | Functional (space)        | Lamé parameter lambda                                           | GPa     | 1.0     |
+--------------------+---------------------------+-----------------------------------------------------------------+---------+---------+
| **eta**            | Functional (space)        | Half the shear-wave impedance for radius damping                | MPa s/m |         |
+--------------------+---------------------------+-----------------------------------------------------------------+---------+---------+
| **sn_pre**         | Functional (space)        | Initial normal stress; Positive in compression                  | MPa     | 0.0     |
+--------------------+---------------------------+-----------------------------------------------------------------+---------+---------+
| **tau_pre**        | Functional (space)        | Initial shear traction                                          | MPa     |         |
+--------------------+---------------------------+-----------------------------------------------------------------+---------+---------+
| **delta_sn**       | Functional (space, time)  | (Optional) Time-dependent change in normal stress               | MPa     | 0.0     |
+--------------------+---------------------------+-----------------------------------------------------------------+---------+---------+
| **delta_tau**      | Functional (space, time)  | (Optional) Time-dependent change in shear traction              | MPa     | 0.0     |
+--------------------+---------------------------+-----------------------------------------------------------------+---------+---------+
| **source**         | Functional (space, time)  | (Optional) Defines an additional source term :math:`s` to the   |         |         |
|                    |                           | state evolution equation :math:`\psi_{,t} = G(V,\psi) + s(x,t)` |         |         |
+--------------------+---------------------------+-----------------------------------------------------------------+---------+---------+
| **fault_solution** | Functional (space, time)  | (Optional) Defines slip and state variable. Useful for          |         |         |
|                    |                           | analytic solutions or manufactured solutions.                   |         |         |
+--------------------+---------------------------+-----------------------------------------------------------------+---------+---------+

For more descriptions about units and scaling, please check out :doc:`Equation scaling section <../reference/scaling>`.


Commented lua script:
---------------------

.. code:: lua

   local Tutorial = {}
   Tutorial.__index = Tutorial

   -- constant parameters
   Tutorial.b = 0.010       -- Rate-and-state evolution effect parameter, b
   Tutorial.V0 = 1.0e-6     -- Reference slip rate [m/s]
   Tutorial.f0 = 0.6        -- Reference friction coefficient

   -- internal parameters
   Tutorial.rho = 2.670     -- Density
   Tutorial.cs = 3.464      -- Shear velocity [km/s]
   Tutorial.nu = 0.25       -- Poisson ratio

   function Tutorial.new(params)
       -- You can define parameters that you may want to change for each scenario
       local self = setmetatable({}, Tutorial)
       self.dip = params.dip
       self.Vp = params.Vp
       return self
   end

   function Tutorial:boundary(x, y, t)
       -- Boundary condition governing Dirichlet boundary in the mesh
       local Vh = self.Vp * t / 2.0
       if x < 0 then
           Vh = -Vh
       end
       return Vh, 0.0
   end

   function Tutorial:mu(x, y)
       -- Shear modulus [GPa]
       return self.cs^2 * self.rho
   end

   function Tutorial:lam(x, y)
       -- Lame parameter lambda [GPa]
       return 2 * self.nu * self:mu(x,y) / (1 - 2 * self.nu)
   end

   function Tutorial:eta(x, y)
       -- Half the shear-wave impedance for radius damping [MPa s/m]
       return self.cs * self.rho / 2.0
   end

   function Tutorial:L(x, y)
       -- Critical distance or state evolution distance (D_RS) [m]
       return 0.008
   end

   function Tutorial:Sinit(x, y)
       -- Initial slip [m]
       return 0.0
   end

   function Tutorial:Vinit(x, y)
       -- Initial slip rate [m/s]
       return self.Vp * math.cos(self.dip * math.pi / 180.0)
   end

   function Tutorial:a(x, y)
       -- Rate-and-state direct effect parameter, a
       local d = math.min(math.abs(y), 32.2)
       return self.b + -5.1115922342571294e-6*d^3 + 0.00029499040079464792*d^2 - 0.003330761720380433*d + 0.0066855943526305008
   end

   function Tutorial:sn_pre(x, y)
       -- Initial normal stress [MPa]
       return 50.0
   end

   function Tutorial:tau_pre(x, y)
       -- Initial shear traction [MPa]
       local Vi = self:Vinit(x, y)
       local sn = self:sn_pre(x, y)
       local amax = self:a(0, -40)
       local e = math.exp((self.f0 + self.b * math.log(self.V0 / math.abs(Vi))) / amax)
       return -(sn * amax * math.asinh((Vi / (2.0 * self.V0)) * e) + self:eta(x, y) * Vi)
   end

   -- Creating various scenarios
   normal = Tutorial.new{dip=60, Vp=1e-9}
   reverse = Tutorial.new{dip=30, Vp=-1e-9}


.. plot::

   import matplotlib.pyplot as plt

   clamp = lambda y: min(abs(y), 32.2)
   a_b = lambda d: -5.1115922342571294e-6*d**3 + 0.00029499040079464792*d**2 - 0.003330761720380433*d + 0.0066855943526305008

   delta = 0.1
   dmax = 40
   d = [i*delta for i in range(int(dmax/delta)+1)]
   fig, ax = plt.subplots()
   ax.axhline(color='grey', lw=1, linestyle='dotted')
   ax.plot(d, [a_b(clamp(dd)) for dd in d])
   ax.set_ylabel('a - b')
   ax.set_xlabel('depth [km]')
   ax.set_box_aspect(0.33)
   plt.show()