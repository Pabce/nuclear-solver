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
def central_potential_reduced_matrix_element(wavefunctions, V0, mu, Ne1, l1, Ne2, l2, integration_limit, integration_steps):
    r = np.linspace(0, integration_limit, integration_steps)
    # We can compute the reduced matrix element as given by Moshinsky:

    rfunc_1 = wavefunctions[Ne1, l1, :]
    rfunc_2 = wavefunctions[Ne2, l2, :]

    pot = V0 * np.exp(-mu * r**2)

    #return np.trapz(rfunc_1 * pot * rfunc_2, r)
    return np.sum(r**2 * rfunc_1 * pot * rfunc_2) * (r[1] - r[0]) # This is faster than trapz (maybe, try again with the real wavefunctions)


@njit(float64(float64[:,:,:,:], float64[:,:,:,:,:,:,:,:], int64, int64, int64, int64, int64, int64, int64, int64, int64), parallel=True)
def central_potential_ls_coupling_basis_matrix_element(cp_red_matrix, moshinsky_brackets,
                                                         Ne1, l1, Ne2, l2, Ne3, l3, Ne4, l4, lamb):
    
    n1 = (Ne1 - l1) // 2
    n2 = (Ne2 - l2) // 2
    n3 = (Ne3 - l3) // 2
    n4 = (Ne4 - l4) // 2

    small_n_max = big_N_max = n1 + n2
    small_l_max = big_L_max = l1 + l2


    mat_el = 0
    for small_n in prange(small_n_max):
        for small_l in prange(small_l_max):
            for big_N in prange(big_N_max):
                for big_L in prange(big_L_max):
                    
                    small_nprime = int(small_n + n3 - n1 + n4 - n2 + 0.5 * (l3 + l4 - l1 - l2))

                    small_Ne = 2 * small_n + small_l
                    small_Ne_prime = 2 * small_nprime + small_l
                    big_Ne = 2 * big_N + big_L


                    # Conditions for Moshinsky bracket to be non-zero 
                    # Also guarantees conservation of parity: (-1)^(l1+l2) = (-1)^(l+L)
                    if (2 * n1 + l1 + 2 * n2 + l2) != 2 * small_l + small_l + 2 * big_N + big_L:
                        continue
                    if (2 * n3 + l3 + 2 * n4 + l4) != 2 * small_nprime + small_l + 2 * big_N + big_L:
                        continue
                        
                    # Ne1, l1, Ne2, l2, Ne1', l1', l2', lamb  (  Ne2' = Ne1 + Ne2 - Ne1'  )
                    coeff = moshinsky_brackets[small_Ne, small_l, big_Ne, big_L, Ne1, l1, l2, lamb]\
                            * moshinsky_brackets[small_Ne_prime, small_l, big_Ne, big_L, Ne3, l3, l4, lamb]

                    central_potential_el = cp_red_matrix[small_Ne, small_l, small_Ne_prime, small_l]

                    mat_el += coeff * central_potential_el

    return mat_el 

# @jit(float64(float64, float64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64), nopython=True)
def central_potential_J_coupling_matrix_element(cp_ls_matrix, Ne1, l1, j1, Ne2, l2, j2, Ne3, l3, j3, Ne4, l4, j4, J):
    mat_el = 0
    lambda_max = 10

    s1 = j1 - l1
    s2 = j2 - l2
    s3 = j3 - l3
    s4 = j4 - l4
    for lamb in range(lambda_max):
        for S in range(0, 1):
            for Sprime in range(0, 1):
                ls_el = cp_ls_matrix[Ne1, Ne1, Ne3, Ne4, l1, l2, l3, l4, lamb]

                wigner_one = lib.NineJSymbol(2*l1, 2*s1, 2*j1, 2*l2, 2*s2, 2*j2, 2*lamb, 2*S, 2*J)
                wigner_two = lib.NineJSymbol(2*l3, 2*s3, 2*j3, 2*l4, 2*s4, 2*j4, 2*lamb, 2*Sprime, 2*J)

                if wigner_one == 0 or wigner_two == 0:
                    continue
                if wigner_one is np.nan or wigner_two is np.nan:
                    continue

                sqrt_one = np.sqrt((2*j1 + 1) * (2*j2 + 1) * (2*lamb + 1) * (2*S + 1))
                sqrt_two = np.sqrt((2*j3 + 1) * (2*j4 + 1) * (2*lamb + 1) * (2*Sprime + 1))

                mat_el += ls_el * wigner_one * wigner_two * sqrt_one * sqrt_two
    
    return mat_el


def set_wavefunctions(Ne_num, l_num, omega, mass, integration_limit, integration_steps):
    # Set some auxilliary parameters for the calculation
    # First, for the central potential matrix elements we need
    # (how big do we need the matrix?)
    small_l_abs_max = big_L_abs_max = Ne_num * 2
    #small_n_abs_max = big_n_abs_max = Ne_num

    # Set the radial wavefunctions
    wavefunctions = np.zeros((Ne_num * 2 + 1, small_l_abs_max + 1, integration_steps), dtype=np.float64)
    r = np.linspace(0, integration_limit, integration_steps)
    for Ne in range(Ne_num * 2):
        for l in range(0, Ne):
            n = (Ne - l)// 2
            wavefunctions[Ne, l, :] = h3d.wavefunction(r, k=n, l=l, omega=omega, mass=mass)

    return wavefunctions


def set_moshinsky_brackets(Ne_num, l_num):
    # Get the full brackets form the Fortran generated values (for now, when they are bigger
    # you may have to pre-slice it)
    _, full_brackets = convert_fortran.get_moshinsky_arrays()
    print(full_brackets.shape)

    # Get the correct slice
    # Ne1, l1, Ne2, l2, Ne1', l1', l2', lamb  (  Ne2' = Ne1 + Ne2 - Ne1'  )
    # (I'm gonna assume lambda is restricted here...)
    
    moshinsky_brackets = full_brackets[0:Ne_num+1, 0:Ne_num+1, 0:l_num+1, 0:l_num+1, 0:Ne_num+1, 0:l_num+1, 0:l_num+1, 0:2*l_num+1]

    return moshinsky_brackets


@njit(float64[:,:,:,:](float64[:,:,:], float64, float64, int64, int64, float64, int64), parallel=True, fastmath=True)
def set_central_potential_reduced_matrix(wavefunctions, V0, mu, Ne_num, l_num, integration_limit, integration_steps):
    # We could do this more carefully and compute just the elements we need, but for now this is fast enough
    small_l_abs_max = big_L_abs_max = l_num * 2
    small_n_abs_max = big_N_abs_max = Ne_num // 2
    
    central_potential_reduced_matrix = np.zeros((Ne_num+1, small_l_abs_max+1, 
                                                    Ne_num+1, big_L_abs_max+1))
    # max n == (Ne_num - l) // 2
    # N + n = n1 + n2 <= Ne_num
    for l1 in prange(small_l_abs_max):
        for n1 in prange(small_n_abs_max - l1):
            for l2 in prange(big_L_abs_max - l1):
                for n2 in prange(big_N_abs_max - l2 - n1):
                    Ne1 = 2*n1 + l1
                    Ne2 = 2*n2 + l2
                    central_potential_reduced_matrix[Ne1, l1, Ne2, l2] =\
                         central_potential_reduced_matrix_element(wavefunctions, V0, mu,
                                    Ne1, l1, Ne2, l2, integration_limit, integration_steps)   

    return central_potential_reduced_matrix



def set_central_potential_ls_coupling_basis_matrix(cp_red_matrix, moshinsky_brackets, Ne_num, l_num):
    lambda_max = 2*l_num

    central_potential_ls_coupling_basis_matrix = np.zeros((Ne_num+1, l_num+1, Ne_num+1,
                                                         l_num+1, Ne_num+1, l_num+1, Ne_num+1, l_num+1, lambda_max+1))
    # Set the central potential (in ls coupling basis) matrix
    for Ne1 in range(Ne_num):
        for l1 in range(l_num):
            for Ne2 in range(Ne_num):
                for l2 in range(l_num):
                    for Ne3 in range(Ne_num):
                        for l3 in range(l_num):
                            for Ne4 in range(Ne_num):
                                for l4 in range(l_num):
                                    for lamb in range(lambda_max):
                                        central_potential_ls_coupling_basis_matrix[Ne1, l1, Ne2, l2, Ne3, l3, Ne4, l4, lamb] =\
                                            central_potential_ls_coupling_basis_matrix_element(cp_red_matrix, moshinsky_brackets,
                                            Ne1, l1, Ne2, l2, Ne3, l3, Ne4, l4, lamb)

    return central_potential_ls_coupling_basis_matrix



# This is horrible python code, I'm just trying to make it work with numba.
# God, please forgive me.
def main(Ne_num, l_num, omega, mass, integration_limit, integration_steps):
    # Set some auxilliary parameters for the calculation
    # First, for the central potential matrix elements we need
    # (how big do we need the matrix?)
    small_l_abs_max = big_l_abs_max = l_num * 2
    small_n_abs_max = big_n_abs_max = Ne_num

    # Set the radial wavefunctions
    wfs = set_wavefunctions(Ne_num, l_num, omega, mass, integration_limit, integration_steps)

    # Set the central potential reduced matrix elements
    start = time.time()
    central_potential_reduced_matrix = set_central_potential_reduced_matrix(wfs, 100.2, 2.1,
                                                Ne_num, l_num, integration_limit, integration_steps)
    print("Time w/ compilation:", time.time() - start)
    start = time.time()
    central_potential_reduced_matrix = set_central_potential_reduced_matrix(wfs, 100.2, 2.1,
                                                Ne_num, l_num, integration_limit, integration_steps)
    print("Time w/out compilation:", time.time() - start)

    # Set the Moshinsky brackets, reading from file
    moshinsky_brackets = set_moshinsky_brackets(Ne_num, l_num)

    # Set the central potential matrix in ls coupling basis
    central_potential_ls_coupling_basis_matrix = set_central_potential_ls_coupling_basis_matrix(central_potential_reduced_matrix,
                                                                                                moshinsky_brackets, Ne_num, l_num)

    cpj = central_potential_J_coupling_matrix_element(central_potential_ls_coupling_basis_matrix, 1, 1, 1.5, 1, 1, 1.5, 1, 1, 1.5, 1, 1, 1.5, 4)
    print(cpj)
    
    # Finally, get the matrix elements in the J coupling basis

    return


    


if __name__ == '__main__':
    # Call the function
    
    main(2, 2, 1, 1, 10, 1000)

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
