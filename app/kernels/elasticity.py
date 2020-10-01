#!/usr/bin/env python3

from yateto import *
import numpy as np

def add(generator, dim, nbf, Nbf, nq, Nq):
    # volume

    J = Tensor('J', (Nq,))
    G = Tensor('G', (dim, dim, Nq))
    lam = Tensor('lam', (Nbf,))
    lam_Q = Tensor('lam_Q', (Nq,))
    mu = Tensor('mu', (Nbf,))
    mu_Q = Tensor('mu_Q', (Nq,))
    W = Tensor('W', (Nq,))
    lam_W_J = Tensor('lam_W_J', (Nq,))
    mu_W_J = Tensor('mu_W_J', (Nq,))
    E_Q = Tensor('E_Q', (Nbf, Nq))
    matE_Q_T = Tensor('matE_Q_T', (Nq, Nbf))
    Dxi_Q = Tensor('Dxi_Q', (Nbf, dim, Nq))
    Dx_Q = Tensor('Dx_Q', Dxi_Q.shape())
    U = Tensor('U', (Nbf, dim))
    Unew = Tensor('Unew', (Nbf, dim))
    A = Tensor('A', (Nbf, dim, Nbf, dim))
    delta = Tensor('delta', (dim, dim), spp=np.identity(dim))
    matMinv = Tensor('matMinv', (Nbf, Nbf))

    generator.add('project_material', [
        lam['p'] <= matMinv['pk'] * lam_Q['q'] * matE_Q_T['qk'] * W['q'],
        mu['p'] <= matMinv['pk'] * mu_Q['q'] * matE_Q_T['qk'] * W['q']
    ])

    generator.add('precomputeVolume', [
        lam_W_J['q'] <= matE_Q_T['qt'] * lam['t'] * J['q'] * W['q'],
        mu_W_J['q'] <= matE_Q_T['qt'] * mu['t'] * J['q'] * W['q']
    ])

    generator.add('Dx_Q', Dx_Q['kiq'] <= G['eiq'] * Dxi_Q['keq'])

    generator.add('volumeOp', Unew['kp'] <= lam_W_J['q'] * Dx_Q['lrq'] * U['lr'] * Dx_Q['kpq'] \
                + mu_W_J['q'] * Dx_Q['kjq'] * (Dx_Q['ljq'] * U['lp'] + Dx_Q['lpq'] * U['lj']))

    generator.add('assembleVolume', A['kplu'] <= lam_W_J['q'] * Dx_Q['luq'] * Dx_Q['kpq'] \
                + mu_W_J['q'] * Dx_Q['kjq'] * (Dx_Q['ljq'] * delta['pu'] + Dx_Q['lpq'] * delta['ju']))

    # surface

    g = [Tensor('g({})'.format(x), (dim, dim, nq)) for x in range(2)]
    n = Tensor('n', (dim, nq))
    nl = Tensor('nl', (nq,))
    w = Tensor('w', (nq,))
    lam_w = [Tensor('lam_w({})'.format(x), (nq,)) for x in range(2)]
    mu_w = [Tensor('mu_w({})'.format(x), (nq,)) for x in range(2)]
    E_q = [Tensor('E_q({})'.format(x), (Nbf, nq)) for x in range(2)]
    matE_q_T = [Tensor('matE_q_T({})'.format(x), (nq, Nbf)) for x in range(2)]
    Dxi_q = [Tensor('Dxi_q({})'.format(x), (Nbf, dim, nq)) for x in range(2)]
    Dx_q = [Tensor('Dx_q({})'.format(x), (Nbf, dim, nq)) for x in range(2)]
    c0 = [Scalar('c0{}'.format(x)) for x in range(2)]
    c1 = [Scalar('c1{}'.format(x)) for x in range(2)]
    c2 = [Scalar('c2{}'.format(x)) for x in range(2)]
    u = [Tensor('u({})'.format(x), (Nbf, dim)) for x in range(2)]
    unew = [Tensor('unew({})'.format(x), (Nbf, dim)) for x in range(2)]
    u_jump = Tensor('u_jump', (dim, nq)) 
    traction_avg = Tensor('traction_avg', (dim, nq)) 
    a = [[Tensor('a({},{})'.format(x, y), (Nbf, dim, Nbf, dim)) for y in range(2)] for x in range(2)]

    generator.addFamily('precomputeSurface', simpleParameterSpace(2), lambda x: [
        lam_w[x]['q'] <= matE_q_T[x]['qt'] * lam['t'] * w['q'],
        mu_w[x]['q'] <= matE_q_T[x]['qt'] * mu['t'] * w['q']
    ])

    def traction(x):
        return lam_w[x]['q'] * Dx_q[x]['lsq'] * u[x]['ls'] * n['pq'] + mu_w[x]['q'] * \
                (Dx_q[x]['ljq'] * u[x]['lp'] * n['jq'] + Dx_q[x]['lpq'] * u[x]['lj'] * n['jq'])

    def tractionTest(x, utilde):
        return lam_w[x]['q'] * Dx_q[x]['kpq'] * utilde['iq'] * n['iq'] + mu_w[x]['q'] * \
                (Dx_q[x]['kjq'] * utilde['pq'] * n['jq'] + Dx_q[x]['kiq'] * utilde['iq'] * n['pq'])

    def surface(x):
        return unew[x]['kp'] <= unew[x]['kp'] + c0[x] * traction_avg['pq'] * E_q[x]['kq'] + \
                                c1[x] * tractionTest(x, u_jump) + \
                                c2[x] * w['q'] * E_q[x]['kq'] * u_jump['pq'] * nl['q']

    generator.addFamily('Dx_q', simpleParameterSpace(2), lambda x: \
        Dx_q[x]['kiq'] <= g[x]['eiq'] * Dxi_q[x]['keq'])

    generator.add('surfaceOp', [
        traction_avg['pq'] <= 0.5 * (traction(0) + traction(1)),
        u_jump['pq'] <= E_q[0]['lq'] * u[0]['lp'] - E_q[1]['lq'] * u[1]['lp'],
        surface(0),
        surface(1)
    ])

    generator.add('surfaceOpBnd', [
        traction_avg['pq'] <= traction(0),
        u_jump['pq'] <= E_q[0]['lq'] * u[0]['lp'],
        surface(0)
    ])

    def assembleSurface(x, y):
        return a[x][y]['kplu'] <= \
            c0[x] * E_q[x]['kq'] * (lam_w[y]['q'] * Dx_q[y]['luq'] * n['pq'] + \
            mu_w[y]['q'] * n['jq'] * (Dx_q[y]['ljq'] * delta['pu'] + Dx_q[y]['lpq'] * delta['ju'])) + \
            c1[y] * E_q[y]['lq'] * (lam_w[x]['q'] * Dx_q[x]['kpq'] * n['uq'] + \
            mu_w[x]['q'] * n['jq'] * (Dx_q[x]['kjq'] * delta['pu'] + Dx_q[x]['kuq'] * delta['jp'])) + \
            c2[abs(y-x)] * delta['pu'] * E_q[y]['lq'] * E_q[x]['kq'] * w['q'] * nl['q']

    generator.addFamily('assembleSurface', simpleParameterSpace(2, 2), assembleSurface)

    # Right-hand side

    b = Tensor('b', (Nbf, dim))
    F = Tensor('F', (dim, Nq))
    generator.add('rhsVolume', b['kp'] <= J['q'] * W['q'] * E_Q['kq'] * F['pq'])

    f = Tensor('f', (dim, nq))
    generator.add('rhsFacet', [
        Dx_q[0]['kiq'] <= g[0]['eiq'] * Dxi_q[0]['keq'],
        b['kp'] <= c1[0] * tractionTest(0, f) + \
            c2[0] * w['q'] * E_q[0]['kq'] * f['pq'] * nl['q']
    ])

    # traction

    traction_avg_proj = Tensor('traction_avg_proj', (dim, nbf))
    slip_proj = Tensor('slip_proj', (dim, nbf))
    minv = Tensor('minv', (nbf, nbf))
    enodal = Tensor('enodal', (nq, nbf))
    enodalT = Tensor('enodalT', (nbf, nq))
    Rnodal = Tensor('Rnodal', (dim, dim, nbf))
    generator.add('traction_avg_proj', [
        traction_avg['pq'] <= 0.5 * (traction(0) + traction(1)),
        traction_avg_proj['pk'] <= Rnodal['rpk'] * traction_avg['rq'] * enodal['ql'] * minv['lk']
    ])

    generator.add('evaluate_slip', f['pq'] <= Rnodal['prl'] * slip_proj['rl'] * enodalT['lq'])
