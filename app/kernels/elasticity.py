#!/usr/bin/env python3

from yateto import *
import numpy as np

def add(generator, dim, nbf, Nbf, nq, Nq, petsc_alignment):
    # volume

    J = Tensor('J', (Nq,))
    Jinv_Q = Tensor('Jinv_Q', (Nq,))
    G = Tensor('G', (dim, dim, Nq))
    lam = Tensor('lam', (Nbf,))
    lam_Q = Tensor('lam_Q', (Nq,))
    mu = Tensor('mu', (Nbf,))
    mu_Q = Tensor('mu_Q', (Nq,))
    rhoInv = Tensor('rhoInv', (Nbf,))
    rhoInv_Q = Tensor('rhoInv_Q', (Nq,))
    W = Tensor('W', (Nq,))
    lam_W_J_Q = Tensor('lam_W_J_Q', (Nq,))
    mu_W_J_Q = Tensor('mu_W_J_Q', (Nq,))
    rhoInv_W_Jinv_Q = Tensor('rhoInv_W_Jinv_Q', (Nq,))
    E_Q = Tensor('E_Q', (Nbf, Nq))
    matE_Q_T = Tensor('matE_Q_T', (Nq, Nbf))
    Dxi_Q = Tensor('Dxi_Q', (Nbf, dim, Nq))
    Dx_Q = Tensor('Dx_Q', Dxi_Q.shape())
    U = Tensor('U', (Nbf, dim), alignStride=petsc_alignment)
    U_Q = Tensor('U_Q', (dim, Nq), alignStride=petsc_alignment)
    Unew = Tensor('Unew', (Nbf, dim), alignStride=petsc_alignment)
    A = Tensor('A', (Nbf, dim, Nbf, dim))
    delta = Tensor('delta', (dim, dim), spp=np.identity(dim))
    M = Tensor('M', (Nbf, Nbf))
    matM = Tensor('matM', (Nbf, Nbf))

    generator.add('massMatrix', M['kl'] <= E_Q['kq'] * W['q'] * J['q'] * E_Q['lq'])

    generator.add('project_material_lhs', matM['kl'] <= \
                matE_Q_T['qk'] * W['q'] * J['q'] * matE_Q_T['ql'])
    generator.add('project_material_rhs', [
        lam['k'] <= lam_Q['q'] * matE_Q_T['qk'] * W['q'] * J['q'],
        mu['k'] <= mu_Q['q'] * matE_Q_T['qk'] * W['q'] * J['q'],
        rhoInv['k'] <= rhoInv_Q['q'] * matE_Q_T['qk'] * W['q'] * J['q']
    ])

    generator.add('precomputeVolume', [
        lam_W_J_Q['q'] <= matE_Q_T['qt'] * lam['t'] * W['q'] * J['q'],
        mu_W_J_Q['q'] <= matE_Q_T['qt'] * mu['t'] * W['q'] * J['q'],
        rhoInv_W_Jinv_Q['q'] <= matE_Q_T['qt'] * rhoInv['t'] * W['q'] * Jinv_Q['q']
    ])

    generator.add('Dx_Q', Dx_Q['kiq'] <= G['eiq'] * Dxi_Q['keq'])

    generator.add('volumeOp', Unew['kp'] <= lam_W_J_Q['q'] * Dx_Q['lrq'] * U['lr'] * Dx_Q['kpq'] \
                + mu_W_J_Q['q'] * Dx_Q['kjq'] * (Dx_Q['ljq'] * U['lp'] + Dx_Q['lpq'] * U['lj']))

    generator.add('assembleVolume', A['kplu'] <= lam_W_J_Q['q'] * Dx_Q['luq'] * Dx_Q['kpq'] \
                + mu_W_J_Q['q'] * Dx_Q['kjq'] * (Dx_Q['ljq'] * delta['pu'] + Dx_Q['lpq'] * delta['ju']))

    # surface

    g = [Tensor('g({})'.format(x), (dim, dim, nq)) for x in range(2)]
    n_q = Tensor('n_q', (dim, nq))
    n_unit_q = Tensor('n_unit_q', (dim, nq))
    nl_q = Tensor('nl_q', (nq,))
    w = Tensor('w', (nq,))
    lam_q = [Tensor('lam_q({})'.format(x), (nq,)) for x in range(2)]
    mu_q = [Tensor('mu_q({})'.format(x), (nq,)) for x in range(2)]
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
    traction_q = Tensor('traction_q', (dim, nq))
    traction_op_q = [Tensor('traction_op_q({})'.format(x), (Nbf, dim, dim, nq)) for x in range(2)]
    Lift = [Tensor('Lift({})'.format(x), (Nbf, dim, Nbf)) for x in range(2)]
    L_q = [Tensor('L_q({})'.format(x), (Nbf, dim, dim, nq)) for x in range(2)]
    Minv = [Tensor('Minv({})'.format(x), (Nbf, Nbf)) for x in range(2)]
    a = [[Tensor('a({},{})'.format(x, y), (Nbf, dim, Nbf, dim)) for y in range(2)] for x in range(2)]

    generator.addFamily('precomputeSurface', simpleParameterSpace(2), lambda x: [
        lam_q[x]['q'] <= matE_q_T[x]['qt'] * lam['t'],
        mu_q[x]['q'] <= matE_q_T[x]['qt'] * mu['t']
    ])

    def traction(x, normal):
        return lam_q[x]['q'] * Dx_q[x]['lsq'] * u[x]['ls'] * normal['pq'] + mu_q[x]['q'] * \
                (Dx_q[x]['ljq'] * u[x]['lp'] * normal['jq'] + Dx_q[x]['lpq'] * u[x]['lj'] * normal['jq'])

    def tractionTest(x, utilde):
        return lam_q[x]['q'] * Dx_q[x]['kpq'] * utilde['iq'] * n_q['iq'] + mu_q[x]['q'] * \
                (Dx_q[x]['kjq'] * utilde['pq'] * n_q['jq'] + Dx_q[x]['kiq'] * utilde['iq'] * n_q['pq'])

    def surface(x):
        return unew[x]['kp'] <= unew[x]['kp'] + c0[x] * traction_q['pq'] * E_q[x]['kq'] * w['q'] + \
                                c1[x] * tractionTest(x, u_jump) * w['q'] + \
                                c2[x] * w['q'] * E_q[x]['kq'] * u_jump['pq'] * nl_q['q']

    generator.addFamily('Dx_q', simpleParameterSpace(2), lambda x: \
        Dx_q[x]['kiq'] <= g[x]['eiq'] * Dxi_q[x]['keq'])

    generator.add('surfaceOp', [
        traction_q['pq'] <= 0.5 * (traction(0, n_q) + traction(1, n_q)),
        u_jump['pq'] <= E_q[0]['lq'] * u[0]['lp'] - E_q[1]['lq'] * u[1]['lp'],
        surface(0),
        surface(1)
    ])

    generator.add('surfaceOpBnd', [
        traction_q['pq'] <= traction(0, n_q),
        u_jump['pq'] <= E_q[0]['lq'] * u[0]['lp'],
        surface(0)
    ])

    def test_normal(x):
        return lam_q[x]['q'] * delta['us'] * n_q['iq'] + mu_q[x]['q'] * \
            (delta['iu'] * n_q['sq'] + delta['is'] * n_q['uq'])

    generator.addFamily('lift_ip', simpleParameterSpace(2),
        lambda x: L_q[x]['lpuq'] <= E_q[x]['lq'] * delta['pu'] * nl_q['q'])
    generator.addFamily('lift_skeleton', simpleParameterSpace(2), lambda x: [
        Lift[0]['lsm'] <= 0.5 * Minv[0]['mo'] * w['q'] * E_q[0]['oq'] * E_q[x]['lq'] * n_q['sq'],
        Lift[1]['lsn'] <= 0.5 * Minv[1]['no'] * w['q'] * E_q[1]['oq'] * E_q[x]['lq'] * n_q['sq'],
        L_q[x]['liuq'] <= 0.5 * ( \
            test_normal(0) * E_q[0]['mq'] * Lift[0]['lsm'] + \
            test_normal(1) * E_q[1]['nq'] * Lift[1]['lsn'] \
        )
    ])
    generator.add('lift_boundary', [
        Lift[0]['lsm'] <= Minv[0]['mo'] * w['q'] * E_q[0]['oq'] * E_q[0]['lq'] * n_q['sq'],
        L_q[0]['liuq'] <= test_normal(0) * E_q[0]['mq'] * Lift[0]['lsm']
    ])

    def assembleTractionOp(x):
        return traction_op_q[x]['kpuq'] <= lam_q[x]['q'] * Dx_q[x]['kpq'] * n_q['uq'] + \
            mu_q[x]['q'] * (n_q['jq'] * Dx_q[x]['kjq'] * delta['pu'] + Dx_q[x]['kuq'] * n_q['pq'])

    def assembleSurface(x, y):
        return a[x][y]['kplu'] <= \
            c0[x] * E_q[x]['kq'] * w['q'] * traction_op_q[y]['lupq'] + \
            c1[y] * E_q[y]['lq'] * w['q'] * traction_op_q[x]['kpuq'] + \
            c2[abs(y-x)] * w['q'] * E_q[x]['kq'] * L_q[y]['lpuq']

    generator.addFamily('assembleTractionOp', simpleParameterSpace(2), assembleTractionOp)
    generator.addFamily('assembleSurface', simpleParameterSpace(2, 2), assembleSurface)

    # Right-hand side


    f_q = Tensor('f_q', (dim, nq))
    f_lifted = [Tensor('f_lifted({})'.format(x), (dim, Nbf, dim)) for x in range(2)]
    f_lifted_q = Tensor('f_lifted_q', (dim, nq))
    generator.add('rhs_lift_ip', f_lifted_q['iq'] <= f_q['iq'] * nl_q['q'])
    generator.add('rhs_lift_boundary', [
        f_lifted[0]['ums'] <= Minv[0]['mo'] * w['q'] * E_q[0]['oq'] * f_q['uq'] * n_q['sq'],
        f_lifted_q['iq'] <= test_normal(0) * E_q[0]['mq'] * f_lifted[0]['ums']
    ])

    generator.add('rhs_lift_skeleton', [
        f_lifted[0]['ums'] <= 0.5 * Minv[0]['mo'] * w['p'] * E_q[0]['op'] * f_q['up'] * n_q['sp'],
        f_lifted[1]['ums'] <= 0.5 * Minv[1]['mo'] * w['p'] * E_q[1]['op'] * f_q['up'] * n_q['sp'],
        f_lifted_q['iq'] <= 0.5 * ( \
            test_normal(0) * E_q[0]['mq'] * f_lifted[0]['ums'] + \
            test_normal(1) * E_q[1]['mq'] * f_lifted[1]['ums'])
    ])

    b = Tensor('b', (Nbf, dim), alignStride=petsc_alignment)
    F_Q = Tensor('F_Q', (dim, Nq))
    generator.add('rhsVolume', b['kp'] <= b['kp'] + J['q'] * W['q'] * E_Q['kq'] * F_Q['pq'])

    generator.add('rhsFacet', [
        Dx_q[0]['kiq'] <= g[0]['eiq'] * Dxi_q[0]['keq'],
        b['kp'] <= b['kp'] + c1[0] * tractionTest(0, f_q) * w['q'] + \
            c2[0] * w['q'] * E_q[0]['kq'] * f_lifted_q['pq']
    ])

    # matrix-free

    U_ext = Tensor('U_ext', (Nbf, dim), alignStride=petsc_alignment)
    u_hat_minus_u_q = Tensor('u_hat_minus_u_q', (nq, dim))
    sigma_hat_q = Tensor('sigma_hat_q', (dim, dim, nq))
    E_Q_T = Tensor('E_Q_T', (Nq, Nbf))
    negative_E_Q = Tensor('negative_E_Q', (Nbf, Nq))
    negative_E_Q_T = Tensor('negative_E_Q_T', (Nq, Nbf))
    E_q_T = [Tensor('E_q_T({})'.format(x), (nq, Nbf)) for x in range(2)]
    negative_E_q = [Tensor('negative_E_q({})'.format(x), (Nbf, nq)) for x in range(2)]
    negative_E_q_T = [Tensor('negative_E_q_T({})'.format(x), (nq, Nbf)) for x in range(2)]
    Ju_Q = Tensor('Ju_Q', (Nq, dim, dim))
    Ju_q = [Tensor('Ju_q({})'.format(x), (nq, dim, dim)) for x in range(2)]
    G_Q_T = Tensor('G_Q_T', (dim, dim, Nq))
    G_q_T = [Tensor('G_q_T({})'.format(x), (dim, dim, nq)) for x in range(2)]
    Dxi_Q_120 = Tensor('Dxi_Q_120', (dim, Nq, Nbf))
    Dxi_q_120 = [Tensor('Dxi_q_120({})'.format(x), (dim, nq, Nbf)) for x in range(2)]
    MinvRef = Tensor('MinvRef', (Nbf, Nbf))


    generator.add('flux_u_skeleton', u_hat_minus_u_q['qi']
            <= 0.5 * (negative_E_q_T[0]['ql'] * U['li'] + E_q_T[1]['ql'] * U_ext['li']))
    generator.add('flux_u_boundary', u_hat_minus_u_q['qi'] <= negative_E_q_T[0]['ql'] * U['li'])

    def constitutive_q(x):
        return lam_q[x]['q'] * delta['ij'] * delta['rs'] * Ju_q[x]['qrs'] \
               + mu_q[x]['q'] * (Ju_q[x]['qij'] + Ju_q[x]['qji'])

    generator.add('flux_sigma_skeleton', [
        Ju_q[0]['qrs'] <= G_q_T[0]['seq'] * Dxi_q_120[0]['eql'] * U['lr'],
        Ju_q[1]['qrs'] <= G_q_T[1]['seq'] * Dxi_q_120[1]['eql'] * U_ext['lr'],
        sigma_hat_q['ijq'] <= 0.5 * (constitutive_q(0) + constitutive_q(1)) +
            c0[0] * (E_q_T[0]['ql'] * U['li'] + negative_E_q_T[1]['ql'] * U_ext['li']) * n_unit_q['jq']
    ])

    generator.add('flux_sigma_boundary', [
        Ju_q[0]['qrs'] <= G_q_T[0]['seq'] * Dxi_q_120[0]['eql'] * U['lr'],
        sigma_hat_q['ijq'] <= constitutive_q(0) + c0[0] * E_q_T[0]['ql'] * U['li'] * n_unit_q['jq']
    ])
    generator.add('apply_volume', [
        Ju_Q['qsr'] <= G_Q_T['seq'] * Dxi_Q_120['eql'] * U['lr'],
        Unew['ku'] <= Dxi_Q['keq'] * G['ejq'] *
            (lam_W_J_Q['q'] * delta['uj'] * delta['rs'] * Ju_Q['qrs'] +
            mu_W_J_Q['q'] * (Ju_Q['quj'] + Ju_Q['qju']))
    ])
    generator.add('apply_facet', Unew['ku'] <= Unew['ku'] + w['q'] * G_q_T[0]['jeq'] * Dxi_q[0]['keq'] *
            (lam_q[0]['q'] * delta['uj'] * u_hat_minus_u_q['qr'] * n_q['rq'] +
            mu_q[0]['q'] * (u_hat_minus_u_q['qu'] * n_q['jq'] + u_hat_minus_u_q['qj'] * n_q['uq'])) +
        w['q'] * negative_E_q[0]['kq'] * n_q['jq'] * sigma_hat_q['ujq']
    )

    generator.add('apply_inverse_mass', Unew['kp'] <=
        MinvRef['kr'] * rhoInv_W_Jinv_Q['q'] * E_Q['rq'] * E_Q['sq'] * MinvRef['sl'] * U['lp'])

    generator.add('project_u_rhs', U['kp'] <= E_Q['kq'] * W['q'] * J['q'] * U_Q['pq'])

    # traction

    generator.add('compute_traction',
        traction_q['pq'] <= 0.5 * (traction(0, n_unit_q) + traction(1, n_unit_q)) +
                            c0[0] * (E_q[0]['lq'] * u[0]['lp'] - E_q[1]['lq'] * u[1]['lp'] - f_q['pq']))

    generator.add('compute_traction_bnd',
        traction_q['pq'] <= traction(0, n_unit_q) +
                            c0[0] * (E_q[0]['lq'] * u[0]['lp'] - f_q['pq']))

