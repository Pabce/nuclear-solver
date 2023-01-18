import numpy as np
from numba import cfunc, jit, float64, int64, int32, int16, int8, void, prange, njit
from numba.experimental import jitclass
import time
#import py3nj


#@cfunc(float64(int64, int64, int64, int64, int64, int64, int64, int64, int64))
@njit(float64(int64, int64, int64, int64, int64, int64, int64, int64, int64))
def get_moshinsky_bracket(n1, n2, l1, l2, n, l, N, L, lamb):
    return 1.0


# This is Minnesota specific, but will live here for now
@njit(float64(float64, float64, int64, int64, int64, int64, float64), fastmath=True)
def central_potential_reduced_matrix_element(V0, mu, n1, n2, l1, l2, integration_limit):
    r = np.linspace(0, integration_limit, 1000)
    # We can compute the reduced matrix element as given by Moshinsky:

    rfunc_1 = r#self.wavefunctions[n1, l1](r)
    rfunc_2 = r#self.wavefunctions[n2, l2](r)

    pot = V0 * np.exp(-mu * r**2)

    #return np.trapz(rfunc_1 * pot * rfunc_2, r)
    return np.sum(rfunc_1 * pot * rfunc_2) * (r[1] - r[0]) # This is faster than trapzv (maybe, try again with the real wavefunctions)


@njit(float64(float64, float64, int64, int64, int64, int64, int64, int64, int64, int64, int64), parallel=True)
def central_potential_ls_coupling_matrix_element(V0, mu, n1, n2, n3, n4, l1, l2, l3, l4, lamb):
    n_max = 10
    l_max = 10
    V0 = 1.
    mu = 1.
    mat_el = 0
    for n in prange(n_max):
        for l in prange(l_max):
            for N in prange(n_max):
                for L in prange(l_max):
                    nprime = int(n + n3 - n1 + n4 - n2 + 0.5 * (l3 + l4 - l1 - l2))
                    # Conditions for Moshinsky bracket to be non-zero 
                    # Also guarantees conservation of parity: (-1)^(l1+l2) = (-1)^(l+L)
                    if (2 * n1 + l1 + 2 * n2 + l2) != 2 * n + l + 2 * N + L:
                        continue
                    if (2 * n3 + l3 + 2 * n4 + l4) != 2 * nprime + l + 2 * N + L:
                        continue

                    coeff = get_moshinsky_bracket(n, l, N, L, n1, n2, l1, l2, lamb)\
                            * get_moshinsky_bracket(nprime, l, N, L, n3, n4, l3, l4, lamb)
                    #central_potential_el = central_potential_reduced_matrix_element(V0, mu, n, l, nprime, l, 10.0)

                    mat_el += coeff * central_potential_reduced_matrix_element(V0, mu, n, l, nprime, l, 10.0)

    return mat_el 

# @jit(float64(float64, float64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64), nopython=True)
# def central_potential_J_coupling_matrix_element(V0, mu, n1, n2, n3, n4, l1, l2, l3, l4, j1, j2, j3, j4, J):
#     mat_el = 0
#     V0 = 1.
#     mu = 1.
#     lambda_max = 10

#     s1 = j1 - l1
#     s2 = j2 - l2
#     s3 = j3 - l3
#     s4 = j4 - l4
#     for lamb in range(lambda_max):
#         for S in range(0, 1):
#             for Sprime in range(0, 1):
#                 ls_el = central_potential_ls_coupling_matrix_element(V0, mu, n1, n2, n3, n4, l1, l2, l3, l4, lamb)
#                 wigner_one = py3nj.wigner9j(2*l1, 2*s1, 2*j1, 2*l2, 2*s2, 2*j2, 2*lamb, 2*S, 2*J)
#                 wigner_two = py3nj.wigner9j(2*l3, 2*s3, 2*j3, 2*l4, 2*s4, 2*j4, 2*lamb, 2*Sprime, 2*J)
#                 sqrt_one = np.sqrt((2*j1 + 1) * (2*j2 + 1) * (2*lamb + 1) * (2*S + 1))
#                 sqrt_two = np.sqrt((2*j3 + 1) * (2*j4 + 1) * (2*lamb + 1) * (2*Sprime + 1))

#                 mat_el += ls_el * wigner_one * wigner_two * sqrt_one * sqrt_two


if __name__ == '__main__':

    start = time.time()
    mel2 = central_potential_ls_coupling_matrix_element(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    #mel2 = central_potential_J_coupling_matrix_element(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    print(mel2)
    print("Time w/ compilation:", time.time() - start)

    start = time.time()
    mel2 = central_potential_ls_coupling_matrix_element(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    #mel2 = central_potential_J_coupling_matrix_element(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    print(mel2)
    print("TIME:", time.time() - start)
