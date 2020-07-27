#!/usr/bin/env python3

from yateto import *
import numpy as np

def add(generator, dim, Nbf, nq, Nq):
    # volume

    J = Tensor('J', (Nq,))
    G = Tensor('G', (dim, dim, Nq))
    lam = Tensor('lam', (Nbf,))
    mu = Tensor('mu', (Nbf,))
    W = Tensor('W', (Nq,))
    lam_W_J = Tensor('lam_W_J', (Nq,))
    mu_W_J = Tensor('mu_W_J', (Nq,))
    E = Tensor('E', (Nbf, Nq))
    Ematerial = Tensor('Ematerial', (Nq, Nbf))
    D_xi = Tensor('D_xi', (Nbf, dim, Nq))
    D_x = Tensor('D_x', D_xi.shape())
    U = Tensor('U', (Nbf, dim))
    Unew = Tensor('Unew', (Nbf, dim))
    A = Tensor('A', (Nbf, dim, Nbf, dim))
    delta = Tensor('delta', (dim, dim), spp=np.identity(dim))

    generator.add('precomputeVolume', [
        lam_W_J['q'] <= Ematerial['qt'] * lam['t'] * J['q'] * W['q'],
        mu_W_J['q'] <= Ematerial['qt'] * mu['t'] * J['q'] * W['q']
    ])

    generator.add('D_x', D_x['kiq'] <= G['eiq'] * D_xi['keq'])

    generator.add('volumeOp', Unew['kp'] <= lam_W_J['q'] * D_x['lrq'] * U['lr'] * D_x['kpq'] \
                        + mu_W_J['q'] * D_x['kjq'] * (D_x['ljq'] * U['lp'] + D_x['lpq'] * U['lj']))

    generator.add('assembleVolume', A['kplu'] <= lam_W_J['q'] * D_x['luq'] * D_x['kpq'] \
                     + mu_W_J['q'] * D_x['kjq'] * (D_x['ljq'] * delta['pu'] + D_x['lpq'] * delta['ju']))

    # surface

    g = [Tensor('g({})'.format(x), (dim, dim, nq)) for x in range(2)]
    n = Tensor('n', (dim, nq))
    nl = Tensor('nl', (nq,))
    w = Tensor('w', (nq,))
    lam_w = [Tensor('lam_w({})'.format(x), (nq,)) for x in range(2)]
    mu_w = [Tensor('mu_w({})'.format(x), (nq,)) for x in range(2)]
    e = [Tensor('e({})'.format(x), (Nbf, nq)) for x in range(2)]
    ematerial = [Tensor('ematerial({})'.format(x), (nq, Nbf)) for x in range(2)]
    d_xi = [Tensor('d_xi({})'.format(x), (Nbf, dim, nq)) for x in range(2)]
    d_x = [Tensor('d_x({})'.format(x), (Nbf, dim, nq)) for x in range(2)]
    c0 = [Scalar('c0{}'.format(x)) for x in range(2)]
    c1 = [Scalar('c1{}'.format(x)) for x in range(2)]
    c2 = [Scalar('c2{}'.format(x)) for x in range(2)]
    u = [Tensor('u({})'.format(x), (Nbf, dim)) for x in range(2)]
    unew = [Tensor('unew({})'.format(x), (Nbf, dim)) for x in range(2)]
    u_jump = Tensor('u_jump', (dim, nq)) 
    traction_avg = Tensor('traction_avg', (dim, nq)) 
    a = [[Tensor('a({},{})'.format(x, y), (Nbf, dim, Nbf, dim)) for y in range(2)] for x in range(2)]

    generator.addFamily('precomputeSurface', simpleParameterSpace(2), lambda x: [
        lam_w[x]['q'] <= ematerial[x]['qt'] * lam['t'] * w['q'],
        mu_w[x]['q'] <= ematerial[x]['qt'] * mu['t'] * w['q']
    ])

    def traction(x):
        return lam_w[x]['q'] * d_x[x]['lsq'] * u[x]['ls'] * n['pq'] + mu_w[x]['q'] * \
                (d_x[x]['ljq'] * u[x]['lp'] * n['jq'] + d_x[x]['lpq'] * u[x]['lj'] * n['jq'])

    def tractionTest(x, utilde):
        return lam_w[x]['q'] * d_x[x]['kpq'] * utilde['iq'] * n['iq'] + mu_w[x]['q'] * \
                (d_x[x]['kjq'] * utilde['pq'] * n['jq'] + d_x[x]['kiq'] * utilde['iq'] * n['pq'])

    def surface(x):
        return unew[x]['kp'] <= unew[x]['kp'] + c0[x] * traction_avg['pq'] * e[x]['kq'] + \
                                c1[x] * tractionTest(x, u_jump) + \
                                c2[x] * w['q'] * e[x]['kq'] * u_jump['pq'] * nl['q']

    generator.addFamily('d_x', simpleParameterSpace(2), lambda x: \
        d_x[x]['kiq'] <= g[x]['eiq'] * d_xi[x]['keq'])

    generator.add('surfaceOp', [
        traction_avg['pq'] <= 0.5 * (traction(0) + traction(1)),
        u_jump['pq'] <= e[0]['lq'] * u[0]['lp'] - e[1]['lq'] * u[1]['lp'],
        surface(0),
        surface(1)
    ])

    generator.add('surfaceOpBnd', [
        traction_avg['pq'] <= traction(0),
        u_jump['pq'] <= e[0]['lq'] * u[0]['lp'],
        surface(0)
    ])

    def assembleSurface(x, y):
        return a[x][y]['kplu'] <= \
            c0[x] * e[x]['kq'] * (lam_w[y]['q'] * d_x[y]['luq'] * n['pq'] + \
               mu_w[y]['q'] * n['jq'] * (d_x[y]['ljq'] * delta['pu'] + d_x[y]['lpq'] * delta['ju'])) + \
            c1[y] * e[y]['lq'] * (lam_w[x]['q'] * d_x[x]['kpq'] * n['uq'] + \
               mu_w[x]['q'] * n['jq'] * (d_x[x]['kjq'] * delta['pu'] + d_x[x]['kuq'] * delta['jp'])) + \
            c2[abs(y-x)] * delta['pu'] * e[y]['lq'] * e[x]['kq'] * w['q'] * nl['q']

    generator.addFamily('assembleSurface', simpleParameterSpace(2, 2), assembleSurface)

    # Right-hand side

    b = Tensor('b', (Nbf, dim))
    F = Tensor('F', (dim, Nq))
    generator.add('rhsVolume', b['kp'] <= J['q'] * W['q'] * E['kq'] * F['pq'])

    f = Tensor('f', (dim, nq))
    generator.add('rhsFacet', [
        d_x[0]['kiq'] <= g[0]['eiq'] * d_xi[0]['keq'],
        b['kp'] <= c1[0] * tractionTest(0, f) + \
            c2[0] * w['q'] * e[0]['kq'] * f['pq'] * nl['q']
    ])

