'''The BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm. 

Key Notes:
1. It is a quasi-Newton method. What is a quasi-Newton method? It is a method that approximates the hessian matrix iteratively, rather than computing it exactly, which is computationally expensive.
2. BFGS approximates the inverse hessian matrix iteratively using gradient information
3. Algorithm:
    1. Start with an initial guess x0
    2. Approximate the hessian matrix using B0
    3. Solve the QP subproblem
        min q(x, s) = f(x) + s^T h(x) + 1/2 s^T H(x)s
        s.t. g(x) + s = 0
    4. Update x = x + alpha s
    5. Update the hessian matrix using BFGS formula
    6. Check for convergence
    7. If not converged, go to step 3


'''