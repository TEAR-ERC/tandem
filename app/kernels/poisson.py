#!/usr/bin/env python3

from yateto import *

def add(generator, dim, nbf, nbf_fault, Nbf, nq, Nq):
    J = Tensor('J', (Nq,))
    G = Tensor('G', (dim, dim, Nq))
    K = Tensor('K', (Nbf,))
    K_Q = Tensor('K_Q', (Nq,))
    W = Tensor('W', (Nq,))
    E = Tensor('E', (Nbf, Nq))
    Em = Tensor('Em', (Nq, Nbf))
    D_xi = Tensor('D_xi', (Nbf, dim, Nq))
    D_x = Tensor('D_x', D_xi.shape())
    A = Tensor('A', (Nbf, Nbf))
    matMinv = Tensor('matMinv', (Nbf, Nbf))
    e_q_T = Tensor('e_q_T', (nq, nbf_fault))
    minv = Tensor('minv', (nbf_fault, nbf_fault))

    generator.add('project_K', K['p'] <= matMinv['pk'] * K_Q['q'] * Em['qk'] * W['q'])

    generator.add('assembleVolume', [
        D_x['kiq'] <= G['eiq'] * D_xi['keq'],
        A['kl'] <= J['q'] * W['q'] * K['m'] * Em['qm'] * D_x['kiq'] * D_x['liq']
    ])

    g = [Tensor('g({})'.format(x), (dim, dim, nq)) for x in range(2)]
    n = Tensor('n', (dim, nq))
    nl = Tensor('nl', (nq,))
    w = Tensor('w', (nq,))
    e = [Tensor('e({})'.format(x), (Nbf, nq)) for x in range(2)]
    em = [Tensor('em({})'.format(x), (nq, Nbf)) for x in range(2)]
    d_xi = [Tensor('d_xi({})'.format(x), (Nbf, dim, nq)) for x in range(2)]
    d_x = [Tensor('d_x({})'.format(x), (Nbf, dim, nq)) for x in range(2)]
    a = [[Tensor('a({},{})'.format(x, y), (Nbf, Nbf)) for y in range(2)] for x in range(2)]
    c0 = [Scalar('c0{}'.format(x)) for x in range(2)]
    c1 = [Scalar('c1{}'.format(x)) for x in range(2)]
    c2 = [Scalar('c2{}'.format(x)) for x in range(2)]

    def surface(x, y):
        return a[x][y]['kl'] <= c0[y] * w['q'] * d_x[x]['kiq'] * n['iq'] * e[y]['lq'] + \
                                c1[x] * w['q'] * d_x[y]['liq'] * n['iq'] * e[x]['kq'] + \
                                c2[abs(y-x)] * w['q'] * e[x]['kq'] * e[y]['lq'] * nl['q']

    surfaceKernelsLocal = [
        d_x[0]['kiq'] <= K['m'] * em[0]['qm'] * g[0]['eiq'] * d_xi[0]['keq'],
        surface(0, 0)
    ]
    surfaceKernelsNeighbour = [
        d_x[1]['kiq'] <= K['m'] * em[1]['qm'] * g[1]['eiq'] * d_xi[1]['keq'],
        surface(0, 1),
        surface(1, 0),
        surface(1, 1)]
    generator.add('assembleFacetLocal', surfaceKernelsLocal)
    generator.add('assembleFacetNeighbour', surfaceKernelsNeighbour)

    b = Tensor('b', (Nbf,))
    F_Q = Tensor('F_Q', (Nq,))
    generator.add('rhsVolume', b['k'] <= b['k'] + J['q'] * W['q'] * E['kq'] * F_Q['q'])

    f_q = Tensor('f_q', (nq,))
    generator.add('rhsFacet',
        b['k'] <= b['k'] + \
            c1[0] * w['q'] * K['m'] * em[0]['qm'] * g[0]['eiq'] * d_xi[0]['keq'] * n['iq'] * f_q['q'] + \
            c2[0] * w['q'] * e[0]['kq'] * nl['q'] * f_q['q'])

    # traction

    def traction(x):
        return lam_w[x]['q'] * d_x[x]['lsq'] * u[x]['ls'] * n['pq'] + mu_w[x]['q'] * \
                (d_x[x]['ljq'] * u[x]['lp'] * n['jq'] + d_x[x]['lpq'] * u[x]['lj'] * n['jq'])

    u = [Tensor('u({})'.format(x), (Nbf,)) for x in range(2)]
    k = [Tensor('k({})'.format(x), (Nbf,)) for x in range(2)]
    grad_u = Tensor('grad_u', (dim, nbf_fault))
    generator.add('grad_u', [
        d_x[0]['kiq'] <= k[0]['m'] * em[0]['qm'] * g[0]['eiq'] * d_xi[0]['keq'],
        d_x[1]['kiq'] <= k[1]['m'] * em[1]['qm'] * g[1]['eiq'] * d_xi[1]['keq'],
        grad_u['pk'] <= 0.5 * (d_x[0]['lpq'] * u[0]['l'] + d_x[1]['lpq'] * u[1]['l']) * w['q'] * \
                        e_q_T['qr'] * minv['rk']
    ])
