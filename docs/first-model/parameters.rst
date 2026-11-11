Parameter file
==============

Tandem simulation parameters are written in the toml script. Here are the key parameters and their descriptions:

- **final_time**: Final simulation time in seconds.
- **mesh_file**: Mesh file.
- **lib**: Lua file containing material & frictional paramters.
- **scenario**: Name of the specific scenario defined in the Lua library.
- **Lua material functions**: Note that the mu, lam, and rho Lua functions receive the volume tag as the last function argument. If region IDs are not set in the mesh, a value of -1 is assumed.
- **type**: Type of problem. Available options: [poisson | elastic/elasticity].
- **mode**: Mode of SEAS simulation. Available options: [QuasiDynamic/QD | QuasiDynamicDiscreteGreen/QDGreen | FullyDynamic/FD].
- **ref_normal**: Define reference normal vector.
- **up**: Define up direction vector.
- **boundary_linear**: Assert that boundary is a linear function of time (i.e. boundary(x, t) = f(x) t). Default = false.
- **matrix_free**: Use matrix-free operators. Default = false.
- **cfl**: CFL tuning parameter (typically <= 1.0). Default = 1.0.
- **mg_coarse_level**: Polynomial degree of coarsest MG level. Default = 1.
- **mg_strategy**: MG level selection strategy. Available options: [TwoLevel | Logarithmic | Full]. Default = TwoLevel.


Computing Green's functions in parallel groups
----------------------------------------------

When using :code:`mode=QDGreen`, every column of the Green's function operator is the traction response to a unit slip and is computed by an independent linear solve. By default all MPI ranks work on one solve at a time. If a single solve no longer scales to the number of ranks in use, the ranks can be split into groups that solve different Green's functions concurrently:

- **gf_groups**: Number of groups the MPI ranks are split into (groups of consecutive ranks). Default = 1, i.e. all ranks work on one Green's function at a time.

For example, running on 144 ranks with :code:`gf_groups = 3` computes three Green's functions at a time, each with 48 ranks.

Every group builds its own copy of the mesh partition, operators and multigrid hierarchy on its sub-communicator, so the memory for these is multiplied by :code:`gf_groups`. The Green's function operator itself is stored only once, distributed over all ranks. The groups work in lock-step and the operator is assembled after every round, so the checkpoints (see below) are the same as without groups. In particular, a run can be restarted with a different value of :code:`gf_groups`. The number of groups must not exceed the number of ranks, and the best value should be found by measuring the time per Green's function for different values.

Distribution of the Green's function operator
---------------------------------------------

- **gf_balance_rows**: Default = true. In :code:`mode=QDGreen` every evaluation of the right-hand side multiplies the dense Green's function operator with the slip. By default the rows of the operator follow the partition of the fault, which concentrates the matrix (and the work of the product) on the ranks that own fault elements, while the other ranks idle. With :code:`gf_balance_rows = true` the rows are distributed evenly over all ranks and the result of the product is sent to the ranks that own the fault elements. This changes only the distribution in memory, not the result or the checkpoint files. Set it to false to recover the previous behaviour.

- **gf_single_precision**: Default = false. The time stepping is limited by the memory bandwidth of the product of the dense Green's function operator with the slip. With :code:`gf_single_precision = true`, the operator is converted to single precision once it is computed or loaded (the checkpoint files stay in double precision) and the double precision copy is freed, which halves the memory traffic and the memory used by the operator. The products are accumulated in double precision, so only the entries of the operator are rounded (relative error about 6e-8). The relative error of a product with a test vector is printed at start-up. Compare with a double precision run before relying on it, in particular if :code:`ts_rtol` is of the order of 1e-7 or smaller.

Green's function operator checkpointing
---------------------------------------

When using :code:`mode=QDGreen`, Green's function checkpointing is parameterized by the **[gf_checkpoint]** section. Here are the key parameters and their descriptions:

- **prefix**: Path where Green's function operator and RHS will be checkpointed.
- **freq_cputime**: CPU time (minutes) frequency between Green's function operator checkpoints.
        
Time stepping checkpointing
---------------------------

Time stepping checkpointing is parameterized by the **[ts_checkpoint]** section. Here are the key parameters and their descriptions:

- **load_directory**: directory from which a checkpoint is loaded. Give path to last_checkpoint.txt to let tandem retrieve the name of the last checkpoint file.
- **save_directory**: Directory where checkpoints are saved. If not specified, it defaults to checkpoint.
- **freq_step**: Time step frequency between checkpoints. This determines how often checkpoints are saved based on the number of time steps. The default value is 1000 steps.
- **freq_cputime**: CPU time frequency between checkpoints, specified in minutes. Checkpoints are saved based on the elapsed CPU time. The default value is 30 minutes.
- **freq_physical_time**: Physical time frequency between checkpoints. This determines how often checkpoints are saved based on the physical time of the simulation. Desactivated by default.
- **storage_type**: Type of storage for checkpoints. It can be one of the following:
    - **none**: Completely deactivate checkpointing.
    - **limited**: Store a finite number of unique checkpoints on disk.
    - **unlimited**: Store all checkpoints without any limit.
- **storage_limited_size**: Number of unique checkpoints stored on disk when storage_type is set to limited. The default value is 2.
 
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

   # Domain outputs
   [domain_output]
   prefix = "output/domain"
   rtol = 0.1

Optionally, you can also specify the format of the output probes (:code:`domain_probe_output` and :code:`fault_probe_output`) in order to be able to use the HDF5 format for instance. To do so, you need to set the type parameter in the probe output section, e.g. :code:`type = "HDF5"`. Note that the HDF5 output format is only available for probe outputs and not for field outputs. For example, 

.. code:: toml

   # On-fault probe outputs
   [fault_probe_output]
   prefix = "fltst_"
   type = "HDF5"
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

Finally, the moment rate writer is also available as an output separately but it is only available in HDF5 format. To use it, you need to use the :code:`[moment_rate_output]` section. You do not need to sepcify a type currently because it is only available in HDF5 format at the moment. But you can specify the prefix for the output files and the frequency of the output using the same parameters as for the other outputs, e.g. :code:`prefix = "output/moment_rate"` and :code:`t_max = 9460800` to output the moment rate every 60 minutes of CPU time. For example,

.. code:: toml

   [moment_rate_output]
   prefix = "output/moment_rate"
   t_max = 9460800
