#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../submodules/yateto/'))

import argparse

from yateto import Tensor, useArchitectureIdentifiedBy, Generator
from yateto.gemm_configuration import GeneratorCollection, Eigen

cmdLineParser = argparse.ArgumentParser()
cmdLineParser.add_argument('--arch', required=True)
cmdLineParser.add_argument('--degree', type=int, required=True)
cmdLineParser.add_argument('--quadPoints', type=int, required=True)
cmdLineParser.add_argument('--outputDir', required=True)
cmdLineArgs = cmdLineParser.parse_args()

arch = useArchitectureIdentifiedBy(cmdLineArgs.arch)

g = Generator(arch)

N = 8
A = Tensor('A', (N, N))
B = Tensor('B', (N, N))
C = Tensor('C', (N, N))

g.add('test', C['ij'] <= A['ik'] * B['kj'])

# Generate code
g.generate(outputDir=cmdLineArgs.outputDir,
           gemm_cfg=GeneratorCollection([Eigen(arch)]),
           namespace='tndm')
