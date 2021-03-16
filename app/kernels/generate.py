#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../submodules/yateto/'))

import argparse
import json

from yateto import Alignment, useArchitectureIdentifiedBy, Generator
from yateto.ast.visitor import PrettyPrinter
from yateto.gemm_configuration import GeneratorCollection, Eigen, LIBXSMM

import poisson
import elasticity
import poisson_adapter
import elasticity_adapter

cmdLineParser = argparse.ArgumentParser()
cmdLineParser.add_argument('--app', required=True)
cmdLineParser.add_argument('--arch', required=True)
cmdLineParser.add_argument('--options', required=True)
cmdLineParser.add_argument('--outputDir', required=True)
cmdLineParser.add_argument('--with_libxsmm', type=bool, default=False)
cmdLineParser.add_argument('--petsc_memalign', type=int, default=8)
cmdLineArgs = cmdLineParser.parse_args()

arch = useArchitectureIdentifiedBy(cmdLineArgs.arch)
petsc_aligned = arch.alignment <= cmdLineArgs.petsc_memalign
petsc_alignment = Alignment.Automatic if petsc_aligned else Alignment.Unaligned
g = Generator(arch)

options = None
with open(cmdLineArgs.options) as j:
    options = json.load(j)

if cmdLineArgs.app == 'poisson':
    poisson.add(g,
            options['dim'],
            options['numFacetBasisFunctions'],
            options['numElementBasisFunctions'],
            options['numFacetQuadPoints'],
            options['numElementQuadPoints'],
            petsc_alignment)
elif cmdLineArgs.app == 'elasticity':
    elasticity.add(g,
            options['degree'],
            options['dim'],
            options['numFacetBasisFunctions'],
            options['numElementBasisFunctions'],
            options['numFacetQuadPoints'],
            options['numElementQuadPoints'],
            petsc_alignment)
elif cmdLineArgs.app == 'poisson_adapter':
    poisson_adapter.add(g,
            options['dim'],
            options['numFaultBasisFunctions'],
            options['numFacetQuadPoints'])
elif cmdLineArgs.app == 'elasticity_adapter':
    elasticity_adapter.add(g,
            options['dim'],
            options['numFaultBasisFunctions'],
            options['numFacetQuadPoints'])

gemmgen_list = []
if cmdLineArgs.with_libxsmm and cmdLineArgs.app == 'elasticity':
    gemmgen_list.append(LIBXSMM(arch))
gemmgen_list.append(Eigen(arch))

# Generate code
g.generate(outputDir=cmdLineArgs.outputDir,
           gemm_cfg=GeneratorCollection(gemmgen_list),
           namespace='tndm::{}'.format(cmdLineArgs.app))

for kernel in g.kernels():
    title = 'AST of {}'.format(kernel.name)
    print(title)
    print('='*len(title))
    for ast in kernel.ast:
        PrettyPrinter().visit(ast)
    print(' ')
