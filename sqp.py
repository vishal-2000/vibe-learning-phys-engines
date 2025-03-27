'''SQP
Sequential Quadratic Programming

Problem:
min f(x)
s.t. h(x) = 0
     g(x) <= 0

Lambda, mu: Lagrange multipliers
Lagrangian: L(x, lambda) = f(x) + lambda^T h(x) + mu^T g(x)
KKT conditions:
1. Gradient of Lagrangian w.r.t. x = 0
2. h(x) = 0
3. g(x) <= 0
4. mu >= 0
5. mu * g(x) = 0

Algorithm:
1. Start with an initial guess x0
2. Solve the QP subproblem
    min q(x, s) = f(x) + s^T h(x) + 1/2 s^T H(x)s
    s.t. g(x) + s = 0
3. Update x = x + alpha s
4. Check for convergence
5. If not converged, go to step 2
'''

import numpy as np
import taichi as ti
import cvxpy as cp

ti.init(arch=ti.gpu)

# Problem definition
def objective(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2

def gradient(x):
    '''Gradient of the objective function'''
    return np.array(2*(x - np.array([1, 2])))

def hessian_approx(B, s, y):
    '''Approximating hessian using BFGS
    '''
    Bs = B @ s
    sy = y.T @ s
    if sy > 1e-6: # Avoiding numerical instability - []
        B += np.outer(y, y) / sy - np.outer(Bs, Bs) / (s.T @ Bs)
    return B