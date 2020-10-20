from yateto import *

def add(generator, dim, nbf_fault, nq):
    e_q_T = Tensor('e_q_T', (nq, nbf_fault))

    slip = Tensor('slip', (nbf_fault,))
    slip_q = Tensor('slip_q', (nq,))

    generator.add('evaluate_slip', slip_q['q'] <= e_q_T['qp'] * slip['p'])

    grad_u = Tensor('grad_u', (dim, nbf_fault))
    unit_normal = Tensor('unit_normal', (dim, nbf_fault))
    traction = Tensor('traction', (nbf_fault,))
    generator.add('evaluate_traction', traction['p'] <= grad_u['kp'] * unit_normal['kp'])
