#!/bin/bash
# An examplary script to run BP6 simulations with 10 cores
# Please update PATH_TO_TANDEM to the path of your own tandem (2d) build

PATH_TO_TANDEM=PATH_TO_YOUR_TANDEM_BUILD/app/tandem

# For BP6-A (dz = 250 m)
mkdir -p outputs_A_250m
mpiexec -bind-to core -n 10 $PATH_TO_TANDEM bp6_A.toml --petsc -options_file options.cfg

# For BP6-S (dz = 50 m)
mkdir -p outputs_S_50m
mpiexec -bind-to core -n 10 $PATH_TO_TANDEM bp6_S.toml --petsc -options_file options.cfg
