#!/usr/bin/env python3

from yateto import *
from math import factorial

def binom(n, k):
    return factorial(n) // factorial(k) // factorial(n - k)

def add(g, dim, degree, Nq):
    Nbf = binom(degree + dim, dim) 
    Dxi = Tensor('Dxi', (Nbf, dim, Nq))
    G = Tensor('G', (dim, dim, Nq))
    detJ = Tensor('detJ', (Nq,))
    w = Tensor('w', (Nq,))
    U = Tensor('U', (Nbf,))
    Unew = Tensor('Unew', (Nbf,))
    Dx = Tensor('Dx', Dxi.shape())
    g.add('volume', [
        Dx['kiq'] <= G['eiq'] * Dxi['keq'],
        Unew['k'] <= detJ['q'] * w['q'] * Dx['kiq'] * Dx['liq'] * U['l']
    ])
