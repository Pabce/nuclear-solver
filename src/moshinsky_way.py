import numpy as np
from numba import cfunc, jit, float64, int64, int32, int16, int8, void, prange, njit, vectorize
from numba.types import unicode_type
import time
from sympy.physics.wigner import wigner_9j
from itertools import product
import pickle

import harmonic_3d as h3d
import convert_fortran

SAVES_DIR = "/Users/pbarham/OneDrive/workspace/cern/hartree/saved_values"

# Load the shared library (Wigner j symbol)
# lib = ctypes.cdll.LoadLibrary('./wigner.so')
# ninej = lib.NineJSymbol
# # Print the lib attributes and functions
# # Declare the function prototype
# # NineJSymbol(double J1, double J2, double J3, double J4, double J5, double J6, double J7, double J8, double J9)
# ninej.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
# ninej.restype = ctypes.c_double


# This is Minnesota specific, but will live here for now
@njit(float64(float64[:,:,:], float64, float64, int64, int64, int64, int64, float64, int64), fastmath=True, cache=True)
def central_potential_reduced_matrix_element(wavefunctions, V0, mu, n1, l1, n2, l2, integration_limit, integration_steps):
    r = np.linspace(0, integration_limit, integration_steps)
    # We can compute the reduced matrix element as given by Moshinsky:

    rfunc_1 = wavefunctions[n1, l1, :]
    rfunc_2 = wavefunctions[n2, l2, :]

    # If rfunc is full of zeros, complain (stupid numba does not support string formatting OR error handling)
    if np.all(rfunc_1 == 0):
        print("rfunc1 is all zeros. n1, l1:", n1, l1)
    if np.all(rfunc_2 == 0):
        print("rfunc2 is all zeros. n2, l2:", n2, l2)

    pot = V0 * np.exp(-mu * r**2)

    return np.trapz(r**2 * rfunc_1 * pot * rfunc_2, r)
    #return np.sum(r**2 * rfunc_1 * pot * rfunc_2) * (r[1] - r[0])  # This is faster than trapz (maybe, try again with the real wavefunctions)



@njit(float64(float64[:,:,:,:], float64[:,:,:,:,:,:,:,:], int64, int64, int64, int64, int64, int64, int64, int64, int64, unicode_type), parallel=True, cache=True)
def central_potential_ls_coupling_basis_matrix_element(cp_red_matrix, moshinsky_brackets,
                                                         n1, l1, n2, l2, n3, l3, n4, l4, lamb, parity_projector):
    # Check lambda is allowed
    if lamb < np.abs(l1 - l2) or lamb > l1 + l2:
        return 0
    if lamb < np.abs(l3 - l4) or lamb > l3 + l4:
        return 0

    Nq_12 = 2 * n1 + l1 + 2 * n2 + l2
    Nq_34 = 2 * n3 + l3 + 2 * n4 + l4 # (I think we just have to consider Nq_12...)
    Nq_lim = Nq_12 # max(Nq_12, Nq_34) # THIS IS NOT Nq_max!!! (the parameter in the function that calls this one, to fill the matrix)


    mat_el = 0
    for small_n in prange(Nq_12 // 2 + 1):
        small_n_prime = small_n + n3 - n1 + n4 - n2 + (l3 + l4 - l1 - l2)//2

        # n' cannot be negative (i.e., if it comes out negative it's because the energy conservation conditions cannot be satisfied)
        if small_n_prime < 0:
            continue

        for small_l in prange(Nq_lim - 2*small_n + 1):
            
            if parity_projector == "even":
                if small_l % 2 != 0:
                    continue
            elif parity_projector == "odd":
                if small_l % 2 != 1:
                    continue

            for big_N in prange(Nq_lim - 2*small_n - small_l + 1):
                for big_L in prange(Nq_lim - 2*small_n - small_l - 2*big_N + 1):
                    

                    # Conditions for Moshinsky bracket to be non-zero 
                    # Also guarantees conservation of parity: (-1)^(l1+l2) = (-1)^(l+L)
                    if (2 * n1 + l1 + 2 * n2 + l2) != 2 * small_n + small_l + 2 * big_N + big_L:
                        continue
                    if (2 * n3 + l3 + 2 * n4 + l4) != 2 * small_n_prime + small_l + 2 * big_N + big_L:
                        continue
                    
                    # print(n1, n2, n3, n4)
                    # print(small_n, small_l, big_N, big_L)

                    # TESTS
                    # m = moshinsky_brackets[small_n_prime, small_l, big_N, big_L, n3, l3, l4, lamb]
                    # print(small_n, small_n_prime, m)
                    # print(small_n_prime, small_l, big_N, big_L, n3, l3, l4, lamb)
                    # wh = np.nonzero(moshinsky_brackets[:,:,:,:,:,0,0,0])
                    # print([wh[i][0] for i in range(3)])

                    # n1, l1, n2, l2, n1', l1', l2', lamb   (  n2' = (2*n1 + l1 + 2*n2 + l2 - 2*n1' - l1' - l2')/2  )
                    coeff = moshinsky_brackets[small_n, small_l, big_N, big_L, n1, l1, l2, lamb]\
                            * moshinsky_brackets[small_n_prime, small_l, big_N, big_L, n3, l3, l4, lamb]
                    
                    # print("n l n' N L", small_n, small_l, small_n_prime, big_N, big_L)
                    # print("Nq_12, Nq_34", Nq_12, Nq_34)
                    # print("coeff", coeff, moshinsky_brackets[small_n, small_l, big_N, big_L, n1, l1, l2, lamb], moshinsky_brackets[small_n_prime, small_l, big_N, big_L, n3, l3, l4, lamb], moshinsky_brackets[small_n_prime, small_l, big_N, big_L, n4, l4, l3, lamb], (-1)**(big_L - lamb))


                    central_potential_el = cp_red_matrix[small_n, small_l, small_n_prime, small_l]
                    #print("central_potential_el", central_potential_el)

                    mat_el += coeff * central_potential_el

    return mat_el 

# 1/2(1 + P_sigma) -> triplet, 1/2(1 - P_sigma) -> singlet, 1/2(1 + P_r) -> even, 1/2(1 - P_r) -> odd
#@njit(float64(float64[:,:,:,:,:,:,:,:,:], int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64, int64), fastmath=True)
def central_potential_J_coupling_matrix_element(cp_ls_matrix, wigner_9j_dict, n1, l1, twoj1, n2, l2, twoj2,
                                                n3, l3, twoj3, n4, l4, twoj4, twoJ, spin_projector, parity_projector):

    twos1, twos2, twos3, twos4 = 1, 1, 1, 1

    # We have included the spin and parity selectors in case we are dealing with a projector in the potential
    # "The overall parity of a many-particle system is the product of the parities of the one-particle states."
    # if parity_projector == "odd": # Select the odd parity states
    #     if (l3 + l4) % 2 == 0:
    #         return 0.
    # elif parity_projector == "even": # Select the even parity states
    #     if (l3 + l4) % 2 == 1:
    #         return 0.

    # TODO: Can you also have 2J, 2J'? -> No, because you only use elements with the same J to compute the final matrix elements
    # Note: in the matrix elements you are asking for, l1=l3, l2=l4, and therefore lambda' is redundant... OR IS IT??
    # Probably, as for central potential matrix elements you only have non-zero if lambda = lambda'
    # In any case, for now...
    if (l1 != l3 or l2 != l4) and (l1 != l4 or l2 != l3):
        return 0.

    mat_el = 0.
    for lamb in prange(np.abs(l1 - l2), l1 + l2 + 1, 1): #for lamb_prime in prange(np.abs(l3 - l4), l3 + l4 + 1, 1):
        for twoS in prange(0, 3, 2):
            for twoS_prime in prange(0, 3, 2):
                # We have included the spin and parity selectors in case we are dealing with a projector in the potential
                if spin_projector == "triplet": # Select the triplet states
                    if twoS_prime == 0:
                        continue
                elif spin_projector == "singlet": # Select the singlet states
                    if twoS_prime == 2 or twoS == 2:
                        continue
                
                # if parity_projector == "odd": # Select the odd parity states
                #     if (lamb+l3+l4) % 2 == 0:
                #         continue
                # elif parity_projector == "even": # Select the even parity states
                #     if (lamb+l3+l4) % 2 == 1:
                #         continue

                ls_el = cp_ls_matrix[n1, l1, n2, l2, n3, l3, n4, l4, lamb]
                # if parity_projector == "odd":
                #     ls_el = 0.5 * (cp_ls_matrix[n1, l1, n2, l2, n3, l3, n4, l4, lamb] - (-1)**(lamb + l3 + l4) * cp_ls_matrix[n1, l1, n2, l2, n4, l4, n3, l3, lamb])
                # elif parity_projector == "even":
                #     ls_el = 0.5 * (cp_ls_matrix[n1, l1, n2, l2, n3, l3, n4, l4, lamb] + (-1)**(lamb + l3 + l4) * cp_ls_matrix[n1, l1, n2, l2, n3, l3, n4, l4, lamb]) 

                # try:
                #     # wigner_one_old = wigner_9j(l1, twos1/2, twoj1/2, l2, twos2/2, twoj2/2, lamb, twoS/2, twoJ/2, prec=64)
                #     # wigner_two_old = wigner_9j(l3, twos3/2, twoj3/2, l4, twos4/2, twoj4/2, lamb, twoS_prime/2, twoJ/2, prec=64)
                # except ValueError:
                #     continue

                wigner_one = float(wigner_9j_dict[(2*l1, twos1, twoj1, 2*l2, twos2, twoj2, 2*lamb, twoS, twoJ)])
                wigner_two = float(wigner_9j_dict[(2*l3, twos3, twoj3, 2*l4, twos4, twoj4, 2*lamb, twoS_prime, twoJ)])

                if wigner_one == 0 or wigner_two == 0:
                    continue

                sqrt_one = np.sqrt((twoj1 + 1) * (twoj2 + 1) * (2*lamb + 1) * (twoS + 1))
                sqrt_two = np.sqrt((twoj3 + 1) * (twoj4 + 1) * (2*lamb + 1) * (twoS_prime + 1))

                mat_el += ls_el * wigner_one * wigner_two * sqrt_one * sqrt_two 
    
    return mat_el


# ----------------------------------------------------------------------------------------------
# BELOW: Functions to set the matrices. This explains the restrictions to l, l'. The above functions are general and should
# work for any l1, l2, l3, l4.
# This is weird... maybe you are wrong about these matrices being diagonal in l, l'


def set_wavefunctions(Ne_max, l_max, hbar_omega, mass, integration_limit, integration_steps):

    # Set the radial wavefunctions
    wavefunctions = np.zeros((Ne_max * 2 + 1, Ne_max * 2 + 1, integration_steps), dtype=np.float64)
    r = np.linspace(0, integration_limit, integration_steps)
    for n in range(Ne_max * 2 + 1):
        for l in range(0, Ne_max * 2 + 1):

            wavefunctions[n, l, :] = h3d.wavefunction(r, k=n, l=l, hbar_omega=hbar_omega, mass=mass)

    return wavefunctions


# TODO: change to dicctionaries, like the wigner_9j_dict (will tell you if you're missing a coefficient)
def set_moshinsky_brackets(Ne_max, l_max):
    # Get the full brackets form the Fortran generated values (for now, when they are bigger
    # you may have to pre-slice it)
    full_brackets, _ = convert_fortran.get_moshinsky_arrays(saves_dir=SAVES_DIR)
    # Get the correct slice
    # n1, l1, n2, l2, n1', l1', l2', lamb   (  n2' = (2*n1 + l1 + 2*n2 + l2 - 2*n1' - l1' - l2')/2  )
    
    #print(full_brackets.shape)
    # The need for these large maxima comes from the calculation of the ls basis elements (this may be wrong)
    # You can probably go way smaller if you check the limits better
    ni_max = Ne_max
    li_max = 2 * Ne_max 

    moshinsky_brackets = full_brackets[0:ni_max+1, 0:li_max+1, 0:ni_max+1, 0:li_max+1, 0:ni_max+1, 0:li_max+1, 0:li_max+1, 0:2*li_max+1]

    return moshinsky_brackets


# load the Wigner 9j dictionary
def set_wigner_9js():
    # Load the pickle file
    with open('{}/wigner_9js.pcl'.format(SAVES_DIR), 'rb') as handle:
        wigner_9j_dict = pickle.load(handle)
    
    return wigner_9j_dict



# This is Minnesota specific
# TODO: do this
@njit(float64[:,:,:,:](float64[:,:,:], float64, float64, int64, int64, float64, int64), parallel=True, fastmath=True, cache=True)
def set_central_potential_reduced_matrix(wavefunctions, V0, mu, Ne_max, l_max, integration_limit, integration_steps):

    Nq_lim = 2 * Ne_max # THIS IS NOT Nq_max!!! (the parameter in the function that calls this one, to fill the matrix)

    # These limits are overkill, but it's fine for now, this does not take much computing time
    central_potential_reduced_matrix = np.zeros((Nq_lim // 2 + 1, Nq_lim + 1, Nq_lim // 2 + 1, Nq_lim + 1), dtype=np.float64) 

    mat_el = 0
    for small_n in prange(Nq_lim // 2 + 1):
        for small_l in prange(Nq_lim + 1): # prange(Nq_lim - 2 * small_n + 1): --> This is the proper limit (the other is just for testing)
            for small_n_prime in prange(Nq_lim // 2 + 1):
            
                central_potential_reduced_matrix[small_n, small_l, small_n_prime, small_l] =\
                        central_potential_reduced_matrix_element(wavefunctions, V0, mu,
                                small_n, small_l, small_n_prime, small_l, integration_limit, integration_steps)   

    return central_potential_reduced_matrix


# TODO: check you are covering all the indices...
@njit(float64[:,:,:,:,:,:,:,:,:](float64[:,:,:,:], float64[:,:,:,:,:,:,:,:], int64, int64, unicode_type), parallel=True, fastmath=True, cache=True)
def set_central_potential_ls_coupling_basis_matrix(cp_red_matrix, moshinsky_brackets, Ne_max, l_max, parity_projector):
    ni_max = Ne_max // 2
    lambda_max = 2*l_max

    central_potential_ls_coupling_basis_matrix = np.zeros((ni_max+1, l_max+1, ni_max+1,
                                                         l_max+1, ni_max+1, l_max+1, ni_max+1, l_max+1, lambda_max+1))
    # Set the central potential (in ls coupling basis) matrix
    for n1 in prange(ni_max + 1):
        l1_0 = min(l_max, Ne_max - 2*n1)
        for l1 in prange(0, l1_0 + 1):
            for n2 in prange(ni_max + 1):
                l2_0 = min(l_max, Ne_max - 2*n2)
                for l2 in prange(0, l2_0 + 1):
                    for n3 in prange(ni_max + 1):
                        l3_0 = min(l_max, Ne_max - 2*n3)
                        for l3 in prange(0, l3_0 + 1):
                            for n4 in prange(ni_max + 1):
                                l4_0 = min(l_max, Ne_max - 2*n4)
                                for l4 in prange(l4_0 + 1):
                                    for lamb in prange(lambda_max + 1):

                                        central_potential_ls_coupling_basis_matrix[n1, l1, n2, l2, n3, l3, n4, l4, lamb] =\
                                            central_potential_ls_coupling_basis_matrix_element(cp_red_matrix, moshinsky_brackets,
                                            n1, l1, n2, l2, n3, l3, n4, l4, lamb, parity_projector)

    return central_potential_ls_coupling_basis_matrix


#@njit(float64[:,:,:,:,:,:,:,:,:,:,:,:,:](float64[:,:,:,:,:,:,:,:,:], int64, int64), parallel=True, fastmath=True)
def set_central_potential_J_coupling_basis_matrix(cp_ls_coupling_matrix, wigner_9j_dict, Ne_max, l_max, spin_projector, parity_projector):
    ni_max = Ne_max // 2
    twoj_max = 2*l_max + 1
    twoJ_max = 2 * twoj_max

    central_potential_J_coupling_basis_matrix = np.zeros((ni_max+1, l_max+1, 2, ni_max+1, l_max+1, 2, ni_max+1, l_max+1, 2, ni_max+1, l_max+1, 2, twoJ_max + 1))

    for n1 in prange(0, ni_max + 1):
        l_0 = min(l_max, Ne_max - 2*n1)
        for l in prange(0, l_0 + 1):
            for twoj in range(np.abs(2*l - 1), 2*l + 2, 2):
                twoj_idx = 0 if twoj < 2*l else 1
                for n2 in prange(0, ni_max + 1):
                    l_prime_0 = min(l_max, Ne_max - 2*n2)
                    for l_prime in prange(0, l_prime_0 + 1):
                        for twoj_prime in range(np.abs(2*l_prime - 1), 2*l_prime + 2, 2):
                            twoj_prime_idx = 0 if twoj_prime < 2*l_prime else 1
                            for n3 in prange(0, (Ne_max - l) // 2 + 1):
                                for n4 in prange(0, (Ne_max - l_prime) // 2 + 1):
                                    for twoJ in range(np.abs(twoj - twoj_prime), twoj + twoj_prime + 1, 2):
                                         
                                        central_potential_J_coupling_basis_matrix[n1, l, twoj_idx, n2, l_prime, twoj_prime_idx,\
                                                                                 n3, l, twoj_idx, n4, l_prime, twoj_prime_idx, twoJ] =\
                                            central_potential_J_coupling_matrix_element(cp_ls_coupling_matrix, wigner_9j_dict, n1, l, twoj, n2, l_prime,
                                                    twoj_prime, n3, l, twoj, n4, l_prime, twoj_prime, twoJ, spin_projector, parity_projector)

                                        central_potential_J_coupling_basis_matrix[n1, l, twoj_idx, n2, l_prime, twoj_prime_idx, \
                                                                                 n4, l_prime, twoj_prime_idx, n3, l, twoj_idx, twoJ] =\
                                            central_potential_J_coupling_matrix_element(cp_ls_coupling_matrix, wigner_9j_dict, n1, l, twoj, n2, l_prime,
                                                    twoj_prime, n4, l_prime, twoj_prime, n3, l, twoj, twoJ, spin_projector, parity_projector)
    
    return central_potential_J_coupling_basis_matrix


@njit(float64[:,:,:,:,:,:,:,:,:,:,:,:](float64[:,:,:,:,:,:,:,:,:,:,:,:,:], int64, int64), parallel=True, fastmath=True, cache=True)
def set_central_potential_matrix(cp_J_coupling_matrix, Ne_max, l_max):
    ni_max = Ne_max // 2
    
    central_potential_basis_matrix = np.zeros((ni_max+1, l_max+1, 2, ni_max+1, l_max+1, 2, ni_max+1, l_max+1, 2, ni_max+1, l_max+1, 2))

    for n1 in prange(0, ni_max + 1):
        l_0 = min(l_max, Ne_max - 2*n1)
        for l in prange(0, l_0 + 1):
            for twoj in range(np.abs(2*l - 1), 2*l + 2, 2):
                twoj_idx = 0 if twoj < 2*l else 1
                for n2 in prange(0, ni_max + 1):
                    l_prime_0 = min(l_max, Ne_max - 2*n2)
                    for l_prime in prange(0, l_prime_0 + 1):
                        for twoj_prime in range(np.abs(2*l_prime - 1), 2*l_prime + 2, 2):
                            twoj_prime_idx = 0 if twoj_prime < 2*l_prime else 1
                            for n3 in prange(0, (Ne_max - l) // 2 + 1):
                                for n4 in prange(0, (Ne_max - l_prime) // 2 + 1):

                                    for twoJ in range(np.abs(twoj - twoj_prime), twoj + twoj_prime + 1, 2):
                                        central_potential_basis_matrix[n1, l, twoj_idx, n2, l_prime, twoj_prime_idx, n3, l, twoj_idx, n4, l_prime, twoj_prime_idx] +=\
                                            cp_J_coupling_matrix[n1, l, twoj_idx, n2, l_prime, twoj_prime_idx, n3, l, twoj_idx, n4, l_prime, twoj_prime_idx, twoJ] *\
                                                (twoJ + 1) / ((twoj + 1) * (twoj_prime + 1))

    return central_potential_basis_matrix

# This is horrible python code, I'm just trying to make it work with numba.
# God, please forgive me.
def main(Ne_max, l_max, hbar_omega, mass, integration_limit, integration_steps):

    # Set the radial wavefunctions
    wfs = set_wavefunctions(Ne_max, l_max, hbar_omega, mass, integration_limit, integration_steps)

    # Set the Moshinsky brackets, reading from file
    start = time.time()
    moshinsky_brackets = set_moshinsky_brackets(Ne_max, l_max)
    print("Time to set Moshinsky brackets:", time.time() - start)

    # Set the Wigner 9j symbols, reading from file
    start = time.time()
    wigner_9j_dict = set_wigner_9js()
    print("Time to set Wigner 9j symbols:", time.time() - start)

    # Set the central potential reduced matrix elements
    start = time.time()
    central_potential_reduced_matrix = set_central_potential_reduced_matrix(wfs, 100.2, 2.1,
                                                Ne_max, l_max, integration_limit, integration_steps)
    print("Time to set reduced matrix elements:", time.time() - start)

    # Set the central potential matrix in ls coupling basis
    start = time.time()
    central_potential_ls_coupling_basis_matrix = set_central_potential_ls_coupling_basis_matrix(central_potential_reduced_matrix,
                                                                                                moshinsky_brackets, Ne_max, l_max)
    print("Time to set ls coupling basis matrix:", time.time() - start)

    n1, n2, n3, n4 = 0, 0, 0, 0
    for n1, n2, n3, n4 in product(range(0, Ne_max // 2 + 1), repeat=4):

        el1 = central_potential_ls_coupling_basis_matrix_element(central_potential_reduced_matrix, moshinsky_brackets,
                                                                        n1, 0, n2, 0, n3, 0, n4, 0, 0)
        el2 = central_potential_ls_coupling_basis_matrix_element(central_potential_reduced_matrix, moshinsky_brackets,
                                                                        n3, 0, n4, 0, n1, 0, n2, 0, 0)

        if not np.isclose(el1, el2):
            print("Error in ls coupling basis matrix element:", n1, n2, n3, n4)
            print(el1, el2)

    # Set the central potential matrix in J coupling basis
    start = time.time()
    central_potential_J_coupling_matrix = set_central_potential_J_coupling_basis_matrix(central_potential_ls_coupling_basis_matrix,
                                                                                                        wigner_9j_dict, Ne_max, l_max)
    print("Time to set J coupling basis matrix:", time.time() - start)
    # cpj = central_potential_J_coupling_matrix_element(central_potential_ls_coupling_basis_matrix, 1, 1, 1.5, 1, 1, 1.5, 1, 1, 1.5, 1, 1, 1.5, 4)
    # print(cpj)
    
    # Get the central potential matrix, once and for all
    start = time.time()
    central_potential_matrix = set_central_potential_matrix(central_potential_J_coupling_matrix, Ne_max, l_max)
    print("Time to set central potential matrix:", time.time() - start)

    return central_potential_matrix


    


if __name__ == '__main__':
    # Call the function
    
    main(Ne_max=8, l_max=0, hbar_omega=3, mass=939, integration_limit=15, integration_steps=2000)

    print("------------------")

    #main(Ne_max=3, l_max=2, hbar_omega=3, mass=1, integration_limit=10, integration_steps=1000)

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
