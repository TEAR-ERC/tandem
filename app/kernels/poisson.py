#!/usr/bin/env python3

from yateto import *

def add(generator, dim, nbf, Nbf, nq, Nq):
    J_Q = Tensor('J_Q', (Nq,))
    Jinv_Q = Tensor('Jinv_Q', (Nq,))
    G_Q = Tensor('G_Q', (dim, dim, Nq))
    K = Tensor('K', (Nbf,))
    K_Q = Tensor('K_Q', (Nq,))
    W = Tensor('W', (Nq,))
    E_Q = Tensor('E_Q', (Nbf, Nq))
    matE_Q_T = Tensor('matE_Q_T', (Nq, Nbf))
    Dxi_Q = Tensor('Dxi_Q', (Nbf, dim, Nq))
    Dx_Q = Tensor('Dx_Q', Dxi_Q.shape())
    A = Tensor('A', (Nbf, Nbf))
    M = Tensor('M', (Nbf, Nbf))
    MinvRef = Tensor('MinvRef', (Nbf, Nbf))
    MinvWA = Tensor('MinvWA', (Nbf, Nbf))
    matM = Tensor('matM', (Nbf, Nbf))

    generator.add('massMatrix', M['kl'] <= E_Q['kq'] * W['q'] * J_Q['q'] * E_Q['lq'])
    generator.add('MinvWA', MinvWA['kl'] <=
        MinvRef['kr'] * W['q'] * Jinv_Q['q'] * E_Q['rq'] * E_Q['sq'] * MinvRef['sl'])

    generator.add('project_K_lhs', matM['kl'] <= matE_Q_T['qk'] * W['q'] * J_Q['q'] * matE_Q_T['ql'])
    generator.add('project_K_rhs', K['k'] <= K_Q['q'] * matE_Q_T['qk'] * W['q'] * J_Q['q'])
    generator.add('K_Q', K_Q['q'] <= K['m'] * matE_Q_T['qm'])

    generator.add('Dx_Q', Dx_Q['kiq'] <= G_Q['eiq'] * Dxi_Q['keq'])
    generator.add('assembleVolume',
        A['kl'] <= J_Q['q'] * W['q'] * K['m'] * matE_Q_T['qm'] * Dx_Q['kiq'] * Dx_Q['liq']
    )

    G_q = Tensor('G_q', (dim, dim, nq))
    n_q = Tensor('n_q', (dim, nq))
    n_unit_q = Tensor('n_unit_q', (dim, nq))
    nl_q = Tensor('nl_q', (nq,))
    w = Tensor('w', (nq,))
    E_q = [Tensor('E_q({})'.format(x), (Nbf, nq)) for x in range(2)]
    matE_q_T = Tensor('matE_q_T', (nq, Nbf))
    Dxi_q = Tensor('Dxi_q', (Nbf, dim, nq))
    K_Dx_q = [Tensor('K_Dx_q({})'.format(x), (Nbf, dim, nq)) for x in range(2)]
    a = [[Tensor('a({},{})'.format(x, y), (Nbf, Nbf)) for y in range(2)] for x in range(2)]
    c0 = [Scalar('c0{}'.format(x)) for x in range(2)]
    c1 = [Scalar('c1{}'.format(x)) for x in range(2)]
    c2 = [Scalar('c2{}'.format(x)) for x in range(2)]
    L_q = [Tensor('L_q({})'.format(x), (Nbf, nq)) for x in range(2)]
    Minv = [Tensor('Minv({})'.format(x), (Nbf, Nbf)) for x in range(2)]
    K_q = [Tensor('K_q({})'.format(x), (nq,)) for x in range(2)]

    generator.add('K_Dx_q', K_Dx_q[0]['kiq'] <= K['m'] * matE_q_T['qm'] * G_q['eiq'] * Dxi_q['keq'])
    generator.add('K_q', K_q[0]['q'] <= K['m'] * matE_q_T['qm'])

    generator.addFamily('lift_ip', simpleParameterSpace(2), \
        lambda x: L_q[x]['lq'] <= E_q[x]['lq'] * nl_q['q'])

    generator.addFamily('lift_skeleton', simpleParameterSpace(2), \
        lambda x: L_q[x]['lq'] <= 0.25 * n_q['iq'] * ( \
            K_q[0]['q'] * E_q[0]['uq'] * \
                Minv[0]['us'] * E_q[0]['sp'] * E_q[x]['lp'] * n_q['ip'] * w['p'] + \
            K_q[1]['q'] * E_q[1]['vq'] * \
                Minv[1]['vs'] * E_q[1]['sp'] * E_q[x]['lp'] * n_q['ip'] * w['p'] \
        ))
    generator.add('lift_boundary', L_q[0]['lq'] <= n_q['iq'] * K_q[0]['q'] * \
        E_q[0]['uq'] * Minv[0]['us'] * E_q[0]['sp'] * E_q[0]['lp'] * n_q['ip'] * w['p'])

    def surface(x, y):
        return a[x][y]['kl'] <= c0[y] * w['q'] * K_Dx_q[x]['kiq'] * n_q['iq'] * E_q[y]['lq'] + \
                                c1[x] * w['q'] * K_Dx_q[y]['liq'] * n_q['iq'] * E_q[x]['kq'] + \
                                c2[abs(y-x)] * w['q'] * E_q[x]['kq'] * L_q[y]['lq']

    generator.addFamily('assembleSurface', simpleParameterSpace(2, 2), surface)

    b = Tensor('b', (Nbf,))
    F_Q = Tensor('F_Q', (Nq,))
    generator.add('rhsVolume', b['k'] <= b['k'] + J_Q['q'] * W['q'] * E_Q['kq'] * F_Q['q'])

    f_q = Tensor('f_q', (nq,))
    f_lifted_q = Tensor('f_lifted_q', (nq,))

    generator.add('rhs_lift_ip', f_lifted_q['q'] <= nl_q['q'] * f_q['q'])

    generator.add('rhs_lift_boundary', f_lifted_q['q'] <= n_q['iq'] * K_q[0]['q'] * \
        E_q[0]['lq'] * Minv[0]['lm'] * E_q[0]['mp'] * f_q['p'] * w['p'] * n_q['ip'])

    generator.add('rhs_lift_skeleton', f_lifted_q['q'] <= 0.25 * n_q['iq'] * ( \
        K_q[0]['q'] * E_q[0]['lq'] * Minv[0]['lm'] * E_q[0]['mp'] * f_q['p'] * w['p'] * n_q['ip'] + \
        K_q[1]['q'] * E_q[1]['lq'] * Minv[1]['lm'] * E_q[1]['mp'] * f_q['p'] * w['p'] * n_q['ip'] \
    ))

    generator.add('rhsFacet', b['k'] <= b['k'] + \
            c1[0] * w['q'] * K_Dx_q[0]['kiq'] * n_q['iq'] * f_q['q'] + \
            c2[0] * w['q'] * E_q[0]['kq'] * f_lifted_q['q'])

    # matrix-free

    sigma = Tensor('sigma', (Nbf, dim))
    U = Tensor('U', (Nbf,))
    U_ext = Tensor('U_ext', (Nbf,))
    U_new = Tensor('U_new', (Nbf,))
    u_hat_q = Tensor('u_hat_q', (nq,))
    sigma_hat_q = Tensor('sigma_hat_q', (dim, nq))
    E_Q_T = Tensor('E_Q_T', (Nq, Nbf))
    E_q_T = [Tensor('E_q_T({})'.format(x), (nq, Nbf)) for x in range(2)]
    MinvRef_E_Q = Tensor('MinvRef_E_Q', (Nbf, Nq))
    K_Jinv_Q = Tensor('K_Jinv_Q', (Nq,))

    generator.add('flux_u_skeleton',
        u_hat_q['q'] <= 0.5 * (E_q_T[0]['ql'] * U['l'] + E_q_T[1]['ql'] * U_ext['l']))
    generator.add('flux_u_boundary', u_hat_q['q'] <= E_q_T[0]['ql'] * U['l'])
    generator.add('stress_volume', sigma['kr'] <= -J_Q['q'] * W['q'] *
        Dxi_Q['keq'] * G_Q['erq'] * E_Q_T['ql'] * U['l'])
    generator.add('stress_facet', sigma['kr'] <= sigma['kr'] +
        w['q'] * E_q[0]['kq'] * n_q['rq'] * u_hat_q['q'])
    generator.add('project_stress', sigma['kr'] <=
        K_Jinv_Q['q'] * W['q'] * MinvRef_E_Q['kq'] * MinvRef_E_Q['lq'] * sigma['lr'])

    generator.add('flux_sigma_skeleton', sigma_hat_q['pq'] <= 0.5 *
            (K_Dx_q[0]['lpq'] * U['l'] + K_Dx_q[1]['lpq'] * U_ext['l']) +
            c0[0] * (E_q_T[0]['ql'] * U['l'] - E_q_T[1]['ql'] * U_ext['l']) * n_unit_q['pq'])
    generator.add('flux_sigma_boundary', sigma_hat_q['pq'] <=
            K_Dx_q[0]['lpq'] * U['l'] + c0[0] * E_q_T[0]['ql'] * U['l'] * n_unit_q['pq'])
    generator.add('apply_volume', U_new['k'] <= W['q'] * J_Q['q'] * E_Q['lq'] *
        G_Q['erq'] * Dxi_Q['keq'] * sigma['lr'])
    generator.add('apply_facet', U_new['k'] <= U_new['k'] -
        w['q'] * E_q[0]['kq'] * n_q['rq'] * sigma_hat_q['rq'])

    # traction

    u = [Tensor('u({})'.format(x), (Nbf,)) for x in range(2)]
    grad_u = Tensor('grad_u', (dim, nq))
    generator.add('grad_u',
        grad_u['pq'] <= 0.5 * (K_Dx_q[0]['lpq'] * u[0]['l'] + K_Dx_q[1]['lpq'] * u[1]['l']) +
                        c0[0] * (E_q[0]['lq'] * u[0]['l'] - E_q[1]['lq'] * u[1]['l'] - f_q['q']) * n_unit_q['pq'])

    generator.add('grad_u_bnd',
        grad_u['pq'] <= K_Dx_q[0]['lpq'] * u[0]['l'] +
                        c0[0] * (E_q[0]['lq'] * u[0]['l'] - f_q['q']) * n_unit_q['pq'])
