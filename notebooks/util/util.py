from sympy import rcode

def lua_code(expr):
    math_funs = ['cos', 'sin', 'sqrt', 'pi', 'exp']
    code = rcode(expr)
    for mf in math_funs:
        code = code.replace(mf, 'math.' + mf)
    return code
