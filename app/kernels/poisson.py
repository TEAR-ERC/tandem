#!/usr/bin/env python3

from yateto import *

def add(g, dim, Nbf, Nq):
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
