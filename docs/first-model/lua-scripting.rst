Lua scripting
=============

.. warning::

   This page is under construction.

.. code:: lua

   local Tutorial = {}
   Tutorial.__index = Tutorial

   -- constant parameters
   Tutorial.b = 0.010
   Tutorial.V0 = 1.0e-6
   Tutorial.f0 = 0.6

   -- internal parameters
   Tutorial.rho = 2.670
   Tutorial.cs = 3.464
   Tutorial.nu = 0.25

   function Tutorial.new(params)
       local self = setmetatable({}, Tutorial)
       self.dip = params.dip
       self.Vp = params.Vp
       return self
   end

   function Tutorial:boundary(x, y, t)
       local Vh = self.Vp * t / 2.0
       if x < 0 then
           Vh = -Vh
       end
       return Vh, 0.0
   end

   function Tutorial:mu(x, y)
       return self.cs^2 * self.rho
   end

   function Tutorial:lam(x, y)
       return 2 * self.nu * self:mu(x,y) / (1 - 2 * self.nu)
   end

   function Tutorial:eta(x, y)
       return self.cs * self.rho / 2.0
   end

   function Tutorial:L(x, y)
       return 0.008
   end

   function Tutorial:Sinit(x, y)
       return 0.0
   end

   function Tutorial:Vinit(x, y)
       return self.Vp * math.cos(self.dip * math.pi / 180.0)
   end

   function Tutorial:a(x, y)
       local d = math.min(math.abs(y), 32.2)
       return self.b + -5.1115922342571294e-6*d^3 + 0.00029499040079464792*d^2 - 0.003330761720380433*d + 0.0066855943526305008
   end

   function Tutorial:sn_pre(x, y)
       return 50.0
   end

   function Tutorial:tau_pre(x, y)
       local Vi = self:Vinit(x, y)
       local sn = self:sn_pre(x, y)
       local amax = self:a(0, -40)
       local e = math.exp((self.f0 + self.b * math.log(self.V0 / math.abs(Vi))) / amax)
       return -(sn * amax * math.asinh((Vi / (2.0 * self.V0)) * e) + self:eta(x, y) * Vi)
   end

   normal = Tutorial.new{dip=60, Vp=1e-9}
   reverse = Tutorial.new{dip=60, Vp=-1e-9}


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

