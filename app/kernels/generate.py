#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../submodules/yateto/'))

import argparse
import json

from yateto import useArchitectureIdentifiedBy, Generator
from yateto.ast.visitor import PrettyPrinter
from yateto.gemm_configuration import GeneratorCollection, Eigen

import poisson
import elasticity

cmdLineParser = argparse.ArgumentParser()
cmdLineParser.add_argument('--app', required=True)
cmdLineParser.add_argument('--arch', required=True)
cmdLineParser.add_argument('--options', required=True)
cmdLineParser.add_argument('--outputDir', required=True)
cmdLineArgs = cmdLineParser.parse_args()

arch = useArchitectureIdentifiedBy(cmdLineArgs.arch)
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
            options['numElementQuadPoints'])
elif cmdLineArgs.app == 'elasticity':
    elasticity.add(g,
            options['dim'],
            options['numFacetBasisFunctions'],
            options['numElementBasisFunctions'],
            options['numFacetQuadPoints'],
            options['numElementQuadPoints'])

# Generate code
g.generate(outputDir=cmdLineArgs.outputDir,
           gemm_cfg=GeneratorCollection([Eigen(arch)]),
           namespace='tndm::{}'.format(cmdLineArgs.app))

for kernel in g.kernels():
    title = 'AST of {}'.format(kernel.name)
    print(title)
    print('='*len(title))
    for ast in kernel.ast:
        PrettyPrinter().visit(ast)
    print(' ')
