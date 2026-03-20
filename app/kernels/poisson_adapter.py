from yateto import *

def add(generator, dim, nbf_fault, nq):
    e_q_T = Tensor('e_q_T', (nq, nbf_fault))
    e_q = Tensor('e_q', (nbf_fault, nq))
    minv = Tensor('minv', (nbf_fault, nbf_fault))
    w = Tensor('w', (nq,))
    mu = Tensor('mu', (nq,))
    nl_q = Tensor('nl_q', (nq, ))
    slip = Tensor('slip', (nbf_fault,))
    slip_q = Tensor('slip_q', (nq,))

    generator.add('evaluate_slip', slip_q['q'] <= e_q_T['qp'] * slip['p'])

    grad_u = Tensor('grad_u', (dim, nq))
    n_q = Tensor('n_q', (dim, nq))
    traction = Tensor('traction', (nbf_fault,))
    generator.add('evaluate_traction', traction['p'] <= minv['rp'] * e_q_T['qr'] * w['q'] * \
                                                        grad_u['kq'] * n_q['kq'])
    slip_rate = Tensor('slip_rate', (nbf_fault, dim))
    slip_rate_q = Tensor('slip_rate_q', (dim, nq))
    moment_rate = Tensor('moment_rate', (dim,))

    generator.add(
        'evaluate_slip_rate', slip_rate_q['pq'] <=
            e_q['lq'] * slip_rate['lp']) 
    generator.add(
        'evaluate_moment_rate', moment_rate['p'] <=
            w['q'] * slip_rate_q['pq'] * nl_q['q'] * mu['q'])