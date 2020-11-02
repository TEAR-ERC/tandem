from yateto import *

def add(generator, dim, nbf_fault, nq):
    e_q_T = Tensor('e_q_T', (nq, nbf_fault))
    minv = Tensor('minv', (nbf_fault, nbf_fault))
    w = Tensor('w', (nq,))

    slip = Tensor('slip', (dim, nbf_fault))
    slip_q = Tensor('slip_q', (dim, nq))

    generator.add('evaluate_slip', slip_q['pq'] <= e_q_T['qk'] * slip['pk'])

    traction_q = Tensor('traction_q', (dim, nq))
    unit_normal = Tensor('unit_normal', (dim, nbf_fault))
    traction = Tensor('traction', (nbf_fault,))
    generator.add('evaluate_traction', traction['p'] <= minv['rp'] * e_q_T['qr'] * w['q'] * \
                                                        traction_q['kq'] * unit_normal['kp'])
