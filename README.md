<h1 align="center">
  tandem
</h1>

<h4 align="center">
	A scalable discontinuous Galerkin code on unstructured curvilinear grids
    for linear elasticity problems and sequences of earthquakes and aseismic slip
 </h4>
 
 <p align="center">
 	<a href="https://opensource.org/licenses/BSD-3-Clause">
    	<img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" alt="License"/>
    </a>
    <a href="https://github.com/TEAR-ERC/tandem/stargazers">
    	<img src="https://img.shields.io/github/stars/TEAR-ERC/tandem" alt="Stars"/>
    </a>
    <a href="https://github.com/TEAR-ERC/tandem/network/members">
    	<img src="https://img.shields.io/github/forks/TEAR-ERC/tandem" alt="Forks"/>
    </a>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#dependencies">Dependencies</a> •
  <a href="#installation">Installation</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#citing-tandem">Citing tandem</a> •
  <a href="#license">License</a>
</p>

![examples](https://raw.githubusercontent.com/TEAR-ERC/tandem/main/docs/images/banner.png)

## Features

* High-order discontinuous Galerkin finite element method
* Solver for elastostatic problems and SEAS problems
* Unstructured triangle and tetrahedral meshes
* Curvilinear representation of boundary
* Native support for [Gmsh](https://gmsh.info/)'s MSH file format version 2 (including high-order meshes)
* Sub-element material parameter resolution
* Efficient matrix-free and assembly kernels for many CPU architectures (using [YATeTo](https://doi.org/10.1145/3406835))
* Scalable on distributed memory architectures; access to many linear solvers via [PETSc](https://petsc.org)
* Supports discrete Green's function for SEAS

For a general overview of the project check out our [presentation at virtual EGU 21](https://tear-erc.github.io/tandem-egu21/).

## Dependencies

tandem relies on the following open-source projects:
* [zlib](http://www.zlib.net/)
* [Eigen](https://eigen.tuxfamily.org)
* [Python](https://www.python.org) with [NumPy](https://numpy.org/)
* [Lua](https://www.lua.org)
* [CMake](https://www.cmake.org)
* [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) and [ParMETIS](http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview)
* MPI (e.g. [OpenMPI](https://www.open-mpi.org/))
* [PETSc](https://www.petsc.org)
* [libxsmm](https://github.com/hfp/libxsmm) (optional)

## Installation

You need a recent C++17 compiler (GCC ≥ 8.0 or clang ≥ 8)) and you need to install
the [dependencies](#dependencies).
Then install using the following commands:

```bash
$ git clone https://github.com/TEAR-ERC/tandem
$ cd tandem/
$ git submodule update --init
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DARCH=hsw -DPOLYNOMIAL_DEGREE=4 -DDOMAIN_DIMENSION=2
$ make -j
```

Here the following options were used:
* CMAKE_C_COMPILER and CMAKE_CXX_COMPILER: Select MPI compiler wrappers
* ARCH: Optimize for Intel Haswell (see [cpu_arch_flags.cmake](https://github.com/TEAR-ERC/tandem/blob/main/cmake/cpu_arch_flags.cmake) for more options)
* POLYNOMIAL_DEGREE: The polynomial degree of the finite element spaces
* DOMAIN_DIMENSION: Set 2 for triangle meshes and 3 for tetrahedral meshes

Check out the detailed [installation guide](https://tandem.readthedocs.io/en/latest/getting-started/installation.html).

## Documentation

[Read the docs.](https://tandem.readthedocs.io/)

## Citing tandem

If you use tandem for your research, please cite the following journal article:

Carsten Uphoff, Dave A May, Alice-Agnes Gabriel, A discontinuous Galerkin method for sequences of earthquakes and aseismic slip on multiple faults using unstructured curvilinear grids, *Geophysical Journal International*, Volume 233, Issue 1, April 2023, Pages 586–626, [https://doi.org/10.1093/gji/ggac467](https://doi.org/10.1093/gji/ggac467)

## License

[BSD 3-Clause License](https://github.com/TEAR-ERC/tandem/blob/main/LICENSE.md)
