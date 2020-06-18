#!/usr/bin/env python3

from yateto import *

def add(g, dim, Nbf, fctNq, Nq):
    Dxi = Tensor('Dxi', (Nbf, dim, Nq))
    G = Tensor('G', (dim, dim, Nq))
    absDetJ = Tensor('absDetJ', (Nq,))
    w = Tensor('w', (Nq,))
    U = Tensor('U', (Nbf,))
    Unew = Tensor('Unew', (Nbf,))
    Dx = Tensor('Dx', Dxi.shape())
    g.add('volume', [
        Dx['kiq'] <= G['eiq'] * Dxi['keq'],
        Unew['k'] <= absDetJ['q'] * w['q'] * Dx['kiq'] * Dx['liq'] * U['l']
    ])

    A = Tensor('A', (Nbf,Nbf))
    g.add('assembly', [
        Dx['kiq'] <= G['eiq'] * Dxi['keq'],
        A['kl'] <= absDetJ['q'] * w['q'] * Dx['kiq'] * Dx['liq']
    ])
