Parameter file
==============

.. warning::

   This page is under construction.

.. code:: toml

   final_time = 47304000000
   mesh_file = "tutorial.msh"
   lib = "tutorial.lua"
   scenario = "normal"
   type = "elasticity"
   ref_normal = [1, 0]
   boundary_linear = true

   [fault_probe_output]
   prefix = "fltst_"
   probes = [
       { name = "dp000", x = [0.0, -0.0] },
       { name = "dp025", x = [1.2500000000000002, -2.1650635094610964] },
       { name = "dp050", x = [2.5000000000000004, -4.330127018922193] },
       { name = "dp075", x = [3.750000000000001, -6.495190528383289] },
       { name = "dp100", x = [5.000000000000001, -8.660254037844386] },
       { name = "dp125", x = [6.250000000000002, -10.825317547305483] },
       { name = "dp150", x = [7.500000000000002, -12.990381056766578] },
       { name = "dp175", x = [8.750000000000002, -15.155444566227676] },
       { name = "dp200", x = [10.000000000000002, -17.32050807568877] },
       { name = "dp250", x = [12.500000000000004, -21.650635094610966] },
       { name = "dp300", x = [15.000000000000004, -25.980762113533157] },
       { name = "dp350", x = [17.500000000000004, -30.31088913245535] },
   ]

   [fault_output]
   prefix = "output/fault"
   rtol = 0.1

   [domain_output]
   prefix = "output/domain"
   rtol = 0.1

