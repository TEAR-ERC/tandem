from yateto import *
from yateto.memory import CSCMemoryLayout


def add(generator, dim, nbf_fault, nq):
    e_q = Tensor('e_q', (nbf_fault, nq))
    e_q_T = Tensor('e_q_T', (nq, nbf_fault))
    fault_basis_q = Tensor('fault_basis_q', (dim, dim, nq))
    minv = Tensor('minv', (nbf_fault, nbf_fault))
    w = Tensor('w', (nq, ))
    nl_q = Tensor('nl_q', (nq, ))
    mu = Tensor('mu', (nq, ))

    copy_slip = Tensor('copy_slip', (dim - 1, dim),
                       spp={(d - 1, d): '1.0'
                            for d in range(1, dim)},
                       memoryLayoutClass=CSCMemoryLayout)
    slip = Tensor('slip', (nbf_fault, dim - 1))
    slip_q = Tensor('slip_q', (dim, nq))

    generator.add(
        'evaluate_slip', slip_q['pq'] <=
            e_q['lq'] * fault_basis_q['poq'] * slip['ln'] * copy_slip['no'])

    traction_q = Tensor('traction_q', (dim, nq))
    traction = Tensor('traction', (nbf_fault, dim))
    generator.add('evaluate_traction', traction['kp'] <= minv['lk'] * e_q_T['ql'] * w['q'] * \
                                          nl_q['q'] * traction_q['oq'] * fault_basis_q['opq'])
    
    slip_rate = Tensor('slip_rate', (nbf_fault, dim))
    slip_rate_q = Tensor('slip_rate_q', (dim, nq))
    moment_rate = Tensor('moment_rate', (dim,))

    generator.add(
        'evaluate_slip_rate', slip_rate_q['pq'] <=
            e_q['lq'] * slip_rate['lp']) 
    generator.add(
        'evaluate_moment_rate', moment_rate['p'] <=
            w['q'] * slip_rate_q['pq'] * nl_q['q'] * mu['q'])
