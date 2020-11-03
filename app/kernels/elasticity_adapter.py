from yateto import *


def add(generator, dim, nbf_fault, nq):
    e_q = Tensor('e_q', (nbf_fault, nq))
    e_q_T = Tensor('e_q_T', (nq, nbf_fault))
    fault_basis_q = Tensor('fault_basis_q', (dim, dim, nq))
    minv = Tensor('minv', (nbf_fault, nbf_fault))
    w = Tensor('w', (nq, ))

    slip = Tensor('slip', (dim, nbf_fault))
    slip_q = Tensor('slip_q', (dim, nq))

    generator.add('evaluate_slip', slip_q['pq'] <= e_q['lq'] * fault_basis_q['poq'] * slip['ol'])

    traction_q = Tensor('traction_q', (dim, nq))
    traction = Tensor('traction', (dim, nbf_fault))
    generator.add('evaluate_traction', traction['pk'] <= minv['lk'] * e_q_T['ql'] * w['q'] * \
                                                         traction_q['oq'] * fault_basis_q['opq'])
