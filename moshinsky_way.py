import numpy as np
from numba import cfunc, jit, float64, int64, int32, int16, int8, void, prange, njit, vectorize
from numba.experimental import jitclass
import time
import harmonic_3d as h3d
import convert_fortran
import wigners
import ctypes

# Load the shared library
lib = ctypes.cdll.LoadLibrary('./wigner.so')
# Print the lib attributes and functions
# Declare the function prototype
# NineJSymbol( double J1, double J2, double J3, double J4, double J5, double J6, double J7, double J8, double J9)
lib.NineJSymbol.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
lib.NineJSymbol.restype = ctypes.c_double



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
    return np.sum(r**2 * rfunc_1 * pot * rfunc_2) * (r[1] - r[0]) # This is faster than trapz (maybe, try again with the real wavefunctions)


@njit(float64(float64[:,:,:,:], float64[:,:,:,:,:,:,:,:], int64, int64, int64, int64, int64, int64, int64, int64, int64), parallel=True)
def central_potential_ls_coupling_basis_matrix_element(cp_red_matrix, moshinsky_brackets,
                                                         n1, l1, n2, l2, n3, l3, n4, l4, lamb):

    Nq_12 = 2 * n1 + l1 + 2 * n2 + l2
    Nq_34 = 2 * n3 + l3 + 2 * n4 + l4
    Nq_lim = max(Nq_12, Nq_34) # THIS IS NOT Nq_max!!! (the parameter in the function that calls this one, to fill the matrix)


    mat_el = 0
    for small_n in prange(Nq_lim // 2 + 1):
        for small_l in prange(Nq_lim // 2 - small_n + 1):
            for big_N in prange(Nq_lim // 2 - small_n - small_l + 1):
                for big_L in prange(Nq_lim // 2 - small_n - small_l - big_N + 1):
                    
                    small_n_prime = int(small_n + n3 - n1 + n4 - n2 + 0.5 * (l3 + l4 - l1 - l2))

                    # Conditions for Moshinsky bracket to be non-zero 
                    # Also guarantees conservation of parity: (-1)^(l1+l2) = (-1)^(l+L)
                    if (2 * n1 + l1 + 2 * n2 + l2) != 2 * small_l + small_l + 2 * big_N + big_L:
                        continue
                    if (2 * n3 + l3 + 2 * n4 + l4) != 2 * small_n_prime + small_l + 2 * big_N + big_L:
                        continue
                        
                    # Ne1, l1, Ne2, l2, Ne1', l1', l2', lamb  (  Ne2' = Ne1 + Ne2 - Ne1'  )
                    coeff = moshinsky_brackets[small_n, small_l, big_N, big_L, n1, l1, l2, lamb]\
                            * moshinsky_brackets[small_n_prime, small_l, big_N, big_L, n3, l3, l4, lamb]

                    central_potential_el = cp_red_matrix[small_n, small_l, small_n_prime, small_l]

                    mat_el += coeff * central_potential_el

    return mat_el 

#@njit(float64(float64[:,:,:,:,:,:,:,:,:], int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64))
def central_potential_J_coupling_matrix_element(cp_ls_matrix, n1, l1, twoj1, n2, l2, twoj2, n3, l3, twoj3, n4, l4, twoj4, J):

    twos1 = twoj1 - 2*l1
    twos2 = twoj2 - 2*l2
    twos3 = twoj3 - 2*l3
    twos4 = twoj4 - 2*l4

    mat_el = 0
    for lamb in range(np.abs(l1 - l2), l1 + l2 + 1, 1):
        for twoS in range(-1, 2, 1):
            for twoS_prime in range(-1, 2, 2):
                ls_el = cp_ls_matrix[n1, l1, n2, l2, n3, l3, n4, l4, lamb]

                wigner_one = lib.NineJSymbol(2*l1, twos1, twoj1, 2*l2, twos2, twoj2, 2*lamb, twoS, 2*J)
                wigner_two = lib.NineJSymbol(2*l3, twos3, twoj3, 2*l4, twos4, twoj4, 2*lamb, twoS_prime, 2*J)

                if wigner_one == 0 or wigner_two == 0:
                    continue
                if wigner_one is np.nan or wigner_two is np.nan:
                    continue

                sqrt_one = np.sqrt((twoj1 + 1) *(twoj2 + 1) * (2*lamb + 1) * (twoS + 1))
                sqrt_two = np.sqrt((twoj3 + 1) * (2*twoj4 + 1) * (2*lamb + 1) * (twoS_prime + 1))

                mat_el += ls_el * wigner_one * wigner_two * sqrt_one * sqrt_two
    
    return mat_el


# ----------------------------------------------------------------------------------------------
# BELOW: Functions to set the matrices. This explains the restrictions to l, l'. The above functions are general and should
# work for any l1, l2, l3, l4.


def set_wavefunctions(Ne_max, l_max, omega, mass, integration_limit, integration_steps):

    # Set the radial wavefunctions
    wavefunctions = np.zeros((Ne_max * 2 + 1, Ne_max * 2 + 1, integration_steps), dtype=np.float64)
    r = np.linspace(0, integration_limit, integration_steps)
    for n in range(Ne_max * 2):
        for l in range(0, Ne_max - 2*n):

            wavefunctions[n, l, :] = h3d.wavefunction(r, k=n, l=l, omega=omega, mass=mass)

    return wavefunctions


def set_moshinsky_brackets(Ne_max, l_max):
    # Get the full brackets form the Fortran generated values (for now, when they are bigger
    # you may have to pre-slice it)
    _, full_brackets = convert_fortran.get_moshinsky_arrays()
    print(full_brackets.shape)

    # Get the correct slice
    # Ne1, l1, Ne2, l2, Ne1', l1', l2', lamb  (  Ne2' = Ne1 + Ne2 - Ne1'  )
    # (I'm gonna assume lambda is restricted here...)
    
    moshinsky_brackets = full_brackets[0:Ne_max+1, 0:Ne_max+1, 0:l_max+1, 0:l_max+1, 0:Ne_max+1, 0:l_max+1, 0:l_max+1, 0:2*l_max+1]

    return moshinsky_brackets

# This is Minnesota specific
# TODO: do this
@njit(float64[:,:,:,:](float64[:,:,:], float64, float64, int64, int64, float64, int64), parallel=True, fastmath=True)
def set_central_potential_reduced_matrix(wavefunctions, V0, mu, Ne_max, l_max, integration_limit, integration_steps):

    Nq_lim = 2 * Ne_max # THIS IS NOT Nq_max!!! (the parameter in the function that calls this one, to fill the matrix)

    # These limits are overkill, but it's fine for now, this does not take much computing time
    central_potential_reduced_matrix = np.zeros((Nq_lim // 2 + 1, Nq_lim // 2 + 1, Nq_lim // 2 + 1, Nq_lim // 2 + 1), dtype=np.float64) 

    mat_el = 0
    for small_n in prange(Nq_lim // 2 + 1):
        for small_l in prange(Nq_lim // 2 - small_n + 1):
            for small_n_prime in prange(Nq_lim // 2 + 1):
            
                central_potential_reduced_matrix[small_n, small_l, small_n_prime, small_l] =\
                        central_potential_reduced_matrix_element(wavefunctions, V0, mu,
                                small_n, small_l, small_n_prime, small_l, integration_limit, integration_steps)   

    return central_potential_reduced_matrix


def set_central_potential_ls_coupling_basis_matrix(cp_red_matrix, moshinsky_brackets, Ne_max, l_max):
    ni_max = Ne_max // 2
    lambda_max = 2*l_max

    central_potential_ls_coupling_basis_matrix = np.zeros((ni_max+1, l_max+1, ni_max+1,
                                                         l_max+1, ni_max+1, l_max+1, ni_max+1, l_max+1, lambda_max+1))
    # Set the central potential (in ls coupling basis) matrix
    for n1 in range(ni_max + 1):
        l1_0 = np.min((l_max, Ne_max - 2*n1))
        for l1 in range(0, l1_0 + 1):
            for n2 in range(ni_max + 1):
                l2_0 = np.min((l_max, Ne_max - 2*n2))
                for l2 in range(0, l2_0 + 1):
                    for n3 in range(ni_max + 1):
                        l3_0 = np.min((l_max, Ne_max - 2*n3))
                        for l3 in range(0, l3_0 + 1):
                            for n4 in range(ni_max + 1):
                                l4_0 = np.min((l_max, Ne_max - 2*n4))
                                for l4 in range(l4_0 + 1):
                                    for lamb in range(lambda_max + 1):
                                        central_potential_ls_coupling_basis_matrix[n1, l1, n2, l2, n3, l3, n4, l4, lamb] =\
                                            central_potential_ls_coupling_basis_matrix_element(cp_red_matrix, moshinsky_brackets,
                                            n1, l1, n2, l2, n3, l3, n4, l4, lamb)

    return central_potential_ls_coupling_basis_matrix


# TODO: reduce matrix size by making twoj-indices only take values 0, 1
def set_central_potential_J_coupling_basis_matrix(cp_ls_coupling_matrix, Ne_max, l_max):
    ni_max = Ne_max // 2
    twoj_max = 2*l_max + 1
    twoJ_max = 2 * twoj_max

    central_potential_J_coupling_basis_matrix = np.zeros((ni_max+1, l_max+1, 2, ni_max+1, l_max+1, 2, ni_max+1, l_max+1, 2, ni_max+1, l_max+1, 2, twoJ_max + 1))

    for n1 in prange(0, ni_max + 1):
        l_0 = np.min((l_max, Ne_max - 2*n1))
        for l in prange(0, l_0 + 1):
            for twoj in prange(np.abs(2*l - 1), 2*l + 2, 2):
                twoj_idx = 0 if twoj < 2*l else 1
                for n2 in prange(0, ni_max + 1):
                    lprime_0 = np.min((l_max, Ne_max - 2*n2))
                    for l_prime in prange(0, lprime_0 + 1):
                        for twoj_prime in prange(np.abs(2*l_prime - 1), 2*l_prime + 2, 2):
                            twoj_prime_idx = 0 if twoj_prime < 2*l_prime else 1
                            for n3 in prange(0, (Ne_max - l) // 2 + 1):
                                for n4 in prange(0, (Ne_max - l_prime) // 2 + 1):
                                    for twoJ in prange(np.abs(twoj - twoj_prime), twoj + twoj_prime + 1, 2):
                                        central_potential_J_coupling_basis_matrix[n1, l, twoj_idx, n2, l_prime, twoj_prime_idx, n3, l, twoj_idx, n4, l_prime, twoj_prime_idx, twoJ] =\
                                            central_potential_J_coupling_matrix_element(cp_ls_coupling_matrix,
                                                    n1, l, twoj, n2, l_prime, twoj_prime, n3, l, twoj, n4, l_prime, twoj_prime, twoJ)
    
    return central_potential_J_coupling_basis_matrix


# TODO: add +1 to the limits of the loops!!
def set_central_potential_matrix(cp_J_coupling_matrix, Ne_max, l_max):
    ni_max = Ne_max // 2
    
    central_potential_basis_matrix = np.zeros((ni_max+1, l_max+1, 2, ni_max+1, l_max+1, 2, ni_max+1, l_max+1, 2, ni_max+1, l_max+1, 2))

    for n1 in prange(0, ni_max + 1):
        l0 = np.min((l_max, Ne_max - 2*n1))
        for l in prange(0, l0 + 1):
            for twoj in prange(np.abs(2*l - 1), 2*l + 2, 2):
                twoj_idx = 0 if twoj < 2*l else 1
                for n2 in prange(0, ni_max + 1):
                    l0_prime = np.min((l_max, Ne_max - 2*n2))
                    for l_prime in prange(0, l0_prime + 1):
                        for twoj_prime in prange(np.abs(2*l_prime - 1), 2*l_prime + 2, 2):
                            twoj_prime_idx = 0 if twoj_prime < 2*l_prime else 1
                            for n3 in prange(0, (Ne_max - l) // 2 + 1):
                                for n4 in prange(0, (Ne_max - l_prime) // 2 + 1):

                                    for twoJ in prange(np.abs(twoj - twoj_prime), twoj + twoj_prime + 1, 2):
                                        central_potential_basis_matrix[n1, l, twoj_idx, n2, l_prime, twoj_prime_idx, n3, l, twoj_idx, n4, l_prime, twoj_prime_idx] +=\
                                            cp_J_coupling_matrix[n1, l, twoj_idx, n2, l_prime, twoj_prime_idx, n3, l, twoj_idx, n4, l_prime, twoj_prime_idx, twoJ] *\
                                                (twoJ + 1) / ((twoj + 1) * (twoj_prime + 1))
    
    return central_potential_basis_matrix

# This is horrible python code, I'm just trying to make it work with numba.
# God, please forgive me.
def main(Ne_max, l_max, omega, mass, integration_limit, integration_steps):

    # Set the radial wavefunctions
    wfs = set_wavefunctions(Ne_max, l_max, omega, mass, integration_limit, integration_steps)

    # Set the central potential reduced matrix elements
    start = time.time()
    central_potential_reduced_matrix = set_central_potential_reduced_matrix(wfs, 100.2, 2.1,
                                                Ne_max, l_max, integration_limit, integration_steps)
    print("Time w/ compilation:", time.time() - start)
    start = time.time()
    central_potential_reduced_matrix = set_central_potential_reduced_matrix(wfs, 100.2, 2.1,
                                                Ne_max, l_max, integration_limit, integration_steps)
    print("Time w/out compilation:", time.time() - start)

    # Set the Moshinsky brackets, reading from file
    moshinsky_brackets = set_moshinsky_brackets(Ne_max, l_max)

    # Set the central potential matrix in ls coupling basis
    central_potential_ls_coupling_basis_matrix = set_central_potential_ls_coupling_basis_matrix(central_potential_reduced_matrix,
                                                                                                moshinsky_brackets, Ne_max, l_max)

    # Set the central potential matrix in J coupling basis
    central_potential_J_coupling_matrix = set_central_potential_J_coupling_basis_matrix(central_potential_ls_coupling_basis_matrix, Ne_max, l_max)
    # cpj = central_potential_J_coupling_matrix_element(central_potential_ls_coupling_basis_matrix, 1, 1, 1.5, 1, 1, 1.5, 1, 1, 1.5, 1, 1, 1.5, 4)
    # print(cpj)
    
    # Get the central potential matrix, once and for all
    set_central_potential_matrix(central_potential_J_coupling_matrix, Ne_max, l_max)

    return


    


if __name__ == '__main__':
    # Call the function
    
    main(Ne_max=2, l_max=2, omega=1, mass=1, integration_limit=10, integration_steps=1000)

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
