import numpy as np
from numba import cfunc, jit, float64, int64, int32, int16, int8, void, prange, njit
from numba.experimental import jitclass
import time
import harmonic_3d as h3d
#import py3nj



#@cfunc(float64(int64, int64, int64, int64, int64, int64, int64, int64, int64))
@njit(float64(int64, int64, int64, int64, int64, int64, int64, int64, int64))
def get_moshinsky_bracket(n1, l1, n2, l2, n, l, N, L, lamb):
    return 1.0



# This is Minnesota specific, but will live here for now
@njit(float64(float64[:,:,:], float64, float64, int64, int64, int64, int64, float64, int64), fastmath=True)
def central_potential_reduced_matrix_element(wavefunctions, V0, mu, n1, l1, n2, l2, integration_limit, integration_steps):
    r = np.linspace(0, integration_limit, integration_steps)
    # We can compute the reduced matrix element as given by Moshinsky:

    rfunc_1 = wavefunctions[n1, l1, :]
    rfunc_2 = wavefunctions[n2, l2, :]

    pot = V0 * np.exp(-mu * r**2)

    #return np.trapz(rfunc_1 * pot * rfunc_2, r)
    return np.sum(rfunc_1 * pot * rfunc_2) * (r[1] - r[0]) # This is faster than trapz (maybe, try again with the real wavefunctions)


#@njit(float64(float64[:,:,:,:], float64[:,:,:,:,:,:,:,:], int64, int64, int64, int64, int64, int64, int64, int64, int64), parallel=True)
def central_potential_ls_coupling_basis_matrix_element(cp_red_matrix, moshinsky_brackets,
                                                         n1, l1, n2, l2, n3, l3, n4, l4, lamb):
    n_max = 10
    l_max = 10


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
                        
                    # TODO - fix this
                    coeff = moshinsky_brackets[n, l, N, L, n1, n2, l1, l2, lamb]\
                            * moshinsky_brackets[nprime, l, N, L, n3, n4, l3, l4, lamb]

                    central_potential_el = cp_red_matrix[n, l, nprime, l]

                    #mat_el += coeff * central_potential_reduced_matrix_element(V0, mu, n, l, nprime, l, 10.0, 1000)

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


def set_wavefunctions(n_num, l_num, omega, mass, integration_limit, integration_steps):
        # Set some auxilliary parameters for the calculation
    # First, for the central potential matrix elements we need
    # (how big do we need the matrix?)
    small_l_abs_max = big_l_abs_max = l_num * 2
    small_n_abs_max = big_n_abs_max = int(2 * n_num + l_num)

    # Set the radial wavefunctions
    wavefunctions = np.zeros((small_n_abs_max, small_l_abs_max, integration_steps), dtype=np.float64)
    r = np.linspace(0, integration_limit, integration_steps)
    for n in range(small_n_abs_max):
        for l in range(small_l_abs_max):
            wavefunctions[n, l, :] = h3d.wavefunction(r, k=n, l=l, omega=omega, mass=mass)

    return wavefunctions



@njit(float64[:,:,:,:](float64[:,:,:], float64, float64, int64, int64, float64, int64), parallel=True, fastmath=True)
def set_central_potential_reduced_matrix(wavefunctions, V0, mu, n_num, l_num, integration_limit, integration_steps):
    # We could do this more carefully and compute just the elements we need, but for now this is fast enough
    small_l_abs_max = big_l_abs_max = l_num * 2
    small_n_abs_max = big_n_abs_max = int(2 * n_num + l_num)
    
    central_potential_reduced_matrix = np.zeros((small_n_abs_max, small_l_abs_max, big_n_abs_max, big_l_abs_max))

    for n1 in prange(small_n_abs_max):
        for l1 in prange(small_l_abs_max):
            for n2 in prange(big_n_abs_max):
                for l2 in prange(big_l_abs_max):
                    central_potential_reduced_matrix[n1, l1, n2, l2] =\
                         central_potential_reduced_matrix_element(wavefunctions, V0, mu,
                                    n1, l1, n2, l2, integration_limit, integration_steps)   

    return central_potential_reduced_matrix


def set_moshinsky_brackets(n1, l1, n2, l2, n3, n4, l3, l4, lamb):
    pass


def set_central_potential_ls_coupling_basis_matrix(cp_red_matrix, moshinsky_brackets, n_num, l_num):
    pass


# This is horrible python code, I'm just trying to make it work with numba.
# God, please forgive me.
def main(n_num, l_num, omega, mass, integration_limit, integration_steps):
    # Set some auxilliary parameters for the calculation
    # First, for the central potential matrix elements we need
    # (how big do we need the matrix?)
    small_l_abs_max = big_l_abs_max = l_num * 2
    small_n_abs_max = big_n_abs_max = int(2 * n_num + l_num)

    # Set the radial wavefunctions
    wfs = set_wavefunctions(n_num, l_num, omega, mass, integration_limit, integration_steps)

    # Set the central potential reduced matrix elements
    start = time.time()
    central_potential_reduced_matrix = set_central_potential_reduced_matrix(wfs, 100.2, 2.1,
                                                n_num, l_num, integration_limit, integration_steps)
    print("Time w/ compilation:", time.time() - start)
    start = time.time()
    central_potential_reduced_matrix = set_central_potential_reduced_matrix(wfs, 100.2, 2.1,
                                                n_num, l_num, integration_limit, integration_steps)
    print("Time w/out compilation:", time.time() - start)

    # Set the Moshinsky brackets, reading from file

    # Set the central potential matrix in ls coupling basis
    
    # Finally, get the matrix elements in the J coupling basis

    return


    


if __name__ == '__main__':
    main(5, 5, 1, 1, 10, 1000)

    # wfs = WAVEFUNCTIONS

    # start = time.time()
    # mel2 = central_potential_ls_coupling_matrix_element(wfs, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    # #mel2 = central_potential_J_coupling_matrix_element(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    # print(mel2)
    # print("Time w/ compilation:", time.time() - start)

    # start = time.time()
    # mel2 = central_potential_ls_coupling_matrix_element(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    # #mel2 = central_potential_J_coupling_matrix_element(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    # print(mel2)
    # print("TIME:", time.time() - start)
