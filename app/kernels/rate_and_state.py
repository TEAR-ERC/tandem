from yateto import *

def add(generator, dim, nbf, nq):
    e_q_T = Tensor('e_q_T', (nq, nbf))

    slip = Tensor('slip', (nbf,))
    slip_q = Tensor('slip_q', (nq,))

    generator.add('evaluate_slip', slip_q['q'] <= e_q_T['qp'] * slip['p'])

    grad_u = Tensor('grad_u', (dim, nbf))
    unit_normal = Tensor('unit_normal', (dim, nbf))
    traction = Tensor('traction', (nbf,))
    generator.add('evaluate_traction', traction['p'] <= grad_u['kp'] * unit_normal['kp'])
