Parameter file
==============

Tandem simulation parameters are written in the toml script. Here are the key parameters and their descriptions:

- **final_time**: Final simulation time in seconds.
- **mesh_file**: Mesh file.
- **lib**: Lua file containing material&frictional paramters.
- **scenario**: Name of the specific scenario defined in the Lua library.
- **type**: Type of problem. Available options: [poisson | elastic/elasticity].
- **mode**: Mode of SEAS simulation. Available options: [QuasiDynamic/QD | QuasiDynamicDiscreteGreen/QDGreen | FullyDynamic/FD].
- **ref_normal**: Define reference normal vector.
- **up**: Define up direction vector.
- **boundary_linear**: Assert that boundary is a linear function of time (i.e. boundary(x, t) = f(x) t). Default = false.
- **matrix_free**: Use matrix-free operators. Default = false.
- **cfl**: CFL tuning parameter (typically <= 1.0). Default = 1.0.
- **mg_coarse_level**: Polynomial degree of coarsest MG level. Default = 1.
- **mg_strategy**: MG level selection strategy. Available options: [TwoLevel | Logarithmic | Full]. Default = TwoLevel.

When using :code:`mode=QDGreen`, you can use Green's function checkpointing feature by defining the following parameters:

- **gf_checkpoint_prefix**: Path where Green's function operator and RHS will be checkpointed.
- **gf_checkpoint_every_nmins**: Time interval, in minutes, at which the Green's function operator data is saved to disk. Default = 30.0.

        
Output configurations
---------------------
Five types of outputs are available: **fault output**, **fault probe output**, **fault scalar output**, **domain output**, and **domain probe output**. 

Common options:

- **prefix**: Output file name prefix.
- **atol**: Absolute tolerance for VMax. Default = 1e-50.
- **rtol**: Relative tolerance for VMax. Default = 0.01.
- **t_min**: Minimum time difference between samples. Default = 0.0.
- **t_max**: Maximum time difference between samples. Default = 365 * 24 * 3600.

For domain output:

- **jacobian**: Output Jacobian. Default = false.

For probe outputs:

- **type**: Define the probe output format. Available options: [Tecplot | CSV]. Default = CSV.
- **probes**: Define the names and locations of the probes.
   - **name**: Station name added to the :code:`prefix` to give the complete output file name.
   - **x**: Location of the probe. Units are consistent with your mesh [km].


Commented parameter file:
-------------------------

.. code:: toml

   final_time = 47304000000      # Final simulation time in seconds.
   mesh_file = "tutorial.msh"    # Mesh file.
   lib = "tutorial.lua"          # Lua file containing material&frictional paramters.
   scenario = "normal"           # Name of the specific scenario defined in the lua library.
   type = "elasticity"           # Type of problem: [poisson | elastic/elasticity].
   ref_normal = [1, 0]           # Reference normal vector.
   boundary_linear = true        # Assert that boundary is a linear function of time (i.e. boundary(x, t) = f(x) t). Default = False.

   # On-fault probe outputs
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
       { name = "dp350", x = [17.500000000000004, -30.31088913245535] }
   ]

   # Fault outputs
   [fault_output]
   prefix = "output/fault"
   rtol = 0.1

   # Domain probe outputs
   [domain_output]
   prefix = "output/domain"
   rtol = 0.1