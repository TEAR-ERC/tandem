SCEC Benchmark Problem BP6-QD-A/S
----
Benchmark problems BP6-QD (-A: aging law; -S: slip law; -C: constant friction) are a set of quasi-dynamic two-dimensional (2D) problems in a whole-space with a 1D fault governed by either velocity-strengthening rate-and-state friction or constant friction subjected to perturbations in effective normal stress due to fluid injection and along-fault pore fluid diffusion.
Detailed description of the BP6-QD problems can be found [here](https://strike.scec.org/cvws/seas/benchmark_descriptions.html)

Here, we provide a set of examples to run BP6-QD-A and BP6-QD-S using tandem.

- **generate_mesh.sh**: examples of command line syntaxes to generate meshes with appropriate cell size 
- **bp6.geo**: meshing script to generate BP6 geometry
- **bp6.lua**: lua file containing material & frictional properties
- **bp6_A.toml** & **bp6_S.toml**: parameter files for BP6-QD-A and BP6-QD-S, respectively
- **options.cfg**: PETSc options for tandem simulation (LU, rk45)
- **run_BP6.sh**: examples of command line syntaxes to run BP6 simulations with mpi

**Important note: Tandem uses Aging law in default. Make sure your compilation is using Slip law before running BP6-QD-S**