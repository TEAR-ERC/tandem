from sympy import rcode, cse

def lua_code(expr):
    math_funs = ['cos', 'sin', 'sqrt', 'pi', 'exp']
    code = rcode(expr)
    for mf in math_funs:
        code = code.replace(mf, 'math.' + mf)
    return code

def lua_code_cse(var_expr_pairs):
    vs, es = zip(*var_expr_pairs)
    e_cse = cse(es)
    for k, v in e_cse[0]:
        print("local {} = {}".format(k, lua_code(v)))
    for k, v in zip(vs, e_cse[1]):
        print("local {} = {}".format(k, lua_code(v)))
