#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../submodules/yateto/'))

import argparse

from yateto import useArchitectureIdentifiedBy, Generator
from yateto.ast.visitor import PrettyPrinter
from yateto.gemm_configuration import GeneratorCollection, Eigen

import poisson

cmdLineParser = argparse.ArgumentParser()
cmdLineParser.add_argument('--arch', required=True)
cmdLineParser.add_argument('--dim', type=int, required=True)
cmdLineParser.add_argument('--degree', type=int, required=True)
cmdLineParser.add_argument('--quadPoints', type=int, required=True)
cmdLineParser.add_argument('--outputDir', required=True)
cmdLineArgs = cmdLineParser.parse_args()

arch = useArchitectureIdentifiedBy(cmdLineArgs.arch)

g = Generator(arch)

poisson.add(g, cmdLineArgs.dim, cmdLineArgs.degree, cmdLineArgs.quadPoints)

# Generate code
g.generate(outputDir=cmdLineArgs.outputDir,
           gemm_cfg=GeneratorCollection([Eigen(arch)]),
           namespace='tndm')

for kernel in g.kernels():
    title = 'AST of {}'.format(kernel.name)
    print(title)
    print('='*len(title))
    for ast in kernel.ast:
        PrettyPrinter().visit(ast)
    print(' ')
