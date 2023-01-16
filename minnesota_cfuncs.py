import numpy as np
from scipy import integrate, LowLevelCallable
from numba import cfunc
from numba.types import intc, CPointer, float64, int32, int64
from numpy.polynomial.hermite import Hermite
from scipy.special import genlaguerre, factorial2, binom

FACTORIAL_LOOKUP = np.array([np.math.factorial(i) for i in range(20)], dtype=np.int64)
HERMITE_POLY_LOOKUP = np.array([Hermite.basis(i) for i in range(40)])


@cfunc(int64(int64))
def fast_factorial(n):
    fact = FACTORIAL_LOOKUP[n]
    return fact
    

@cfunc(int64(int64))
def double_factorial(n):
    if n % 2 == 0:
        k = n // 2
        return 2**k * fast_factorial(k)
    else:
        k = (n - 1) // 2
        return fast_factorial(2*k) // (2**k * fast_factorial(k))


@cfunc(float64(float64, float64, float64, float64))
def potential_l0(V0, mu, x1, x2):
    return - 0.5 * V0 / (2 * mu * x1 * x2) * np.exp(-mu * (x1 + x2)**2) * (-1 + np.exp(4 * mu * x1 * x2))


# @cfunc(float64(float64, int32, float64))
# def gen_laguerre(x, n, alpha):
#     if alpha == -0.5:
#         return (-1)**n / (fast_factorial(n) * 2**n) * HERMITE_POLY_LOOKUP[2*n](np.sqrt(x))
#     else:
#         return 1/x * ((x - n) * gen_laguerre(x, n, alpha - 1) + (alpha + n) * gen_laguerre(x, n - 1, alpha - 1))

# @cfunc(int64(int64, int64))
# def gen_binomial(n, k):
#     return binom(n, k)


# @cfunc(float64(float64, int32, float64))
# def gen_laguerre(r, n, alpha):
#     val = 0
#     for i in range(n+1):
#         val += (-1)**i * gen_binomial(n + alpha, n - i) * r**i / fast_factorial(i)
#     return val




# if __name__ == '__main__':
#     for k in range(0, 3):
#         for l in range(0, 3):
#             wf, norm = series_wavefunction(np.linspace(0, 10, 1000), k, l, 10, 1)
#             print(f'k={k}, l={l}, norm={norm}')




