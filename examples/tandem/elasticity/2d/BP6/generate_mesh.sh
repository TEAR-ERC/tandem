#!/bin/bash
# Script to generate mesh for BP6 simulation
# For gmsh, see http://gmsh.info/#Download

# For BP6-A (dz = 250 m)
gmsh -2 bp6.geo -order 2 -setnumber hf 250 -o bp6_hf250.msh
# For BP6-S (dz = 50 m)
gmsh -2 bp6.geo -order 2 -setnumber hf 50 -o bp6_hf50.msh