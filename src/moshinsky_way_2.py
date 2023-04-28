import numpy as np
from numba import cfunc, jit, float64, int64, int32, int16, int8, void, prange, njit, vectorize, boolean
from numba.types import unicode_type, Tuple, DictType, UniTuple
from numba.typed import Dict
import time
from sympy.physics.wigner import wigner_9j
from itertools import product
import pickle
import os
import matplotlib.pyplot as plt

import harmonic_3d as h3d
import convert_fortran

#os.environ['NUMBA_DISABLE_JIT'] = '1'

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
@njit(float64(float64[:,:,:], float64, float64, int64, int64, int64, int64, float64, int64), fastmath=False, cache=True)
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

    # TODO: there seems to be accumulating numerical error. Check how to do it with Talmi integrals.
    b = np.sqrt(197.3269788**2/(10 * 939))
    val = np.trapz(r**2 * rfunc_1 * pot * rfunc_2, r)
    
    # if val < 0:
    #     print("NEG", val, n1, l1, n2, l2)

    return val
    #return np.sum(r**2 * rfunc_1 * pot * rfunc_2) * (r[1] - r[0])  # This is faster than trapz (maybe, try again with the real wavefunctions)



# Plain central potential V(r)
@njit(float64(float64[:,:,:,:], int64, int64, int64, int64, int64, int64, int64,
                                int64, int64, int64, int64, int64, int64), cache=True)
def central_potential_moshinsky_basis_matrix_element(cp_red_matrix, small_n_prime, small_l_prime, big_N_prime, big_L_prime, 
                                                     lamb_prime, twoS_prime, small_n, small_l, big_N, big_L, lamb, twoS, twoJ):
    
    # Deltas
    if small_l_prime != small_l or big_N_prime != big_N or big_L_prime != big_L:
        return 0.
    if lamb_prime != lamb or twoS_prime != twoS:
        return 0.
    
    # J, lambda and S independent
    mat_el = cp_red_matrix[small_n_prime, small_l, small_n, small_l]
    return mat_el 


# Central potential * spin-spin interaction -> (sigma_1 * sigma_2) V(r)
@njit(float64(float64[:,:,:,:], int64, int64, int64, int64, int64, int64, int64,
                                int64, int64, int64, int64, int64, int64), cache=True)
def cp_spin_spin_moshinsky_basis_matrix_element(cp_red_matrix, small_n_prime, small_l_prime, big_N_prime, big_L_prime, lamb_prime, 
                                                twoS_prime, small_n, small_l, big_N, big_L, lamb, twoS, twoJ):
    
    # Deltas (what happens to the delta_M,M' term? How does it enter?)
    if small_l_prime != small_l or big_N_prime != big_N or big_L_prime != big_L:
        return 0.
    if lamb_prime != lamb or twoS_prime != twoS:
        return 0.
    
    # J, lambda independent
    S = twoS // 2

    #print(cp_red_matrix[small_n_prime, small_l, small_n, small_l], "cpred")
    # TODO: why does this not cancel out later when multiplying by 1? (instead of the matrix element)
    mat_el =  2 * (S * (S + 1) - 3/2) * cp_red_matrix[small_n_prime, small_l, small_n, small_l]
    
    return mat_el


#DictType(UniTuple(int64, 9), float64),
@njit(float64(float64[:,:,:,:, :,:,:], float64[:,:,:,:, :,:,:,:], DictType(UniTuple(int64, 9), float64),
                                int64, int64, int64, int64, int64, int64,
                                int64, int64, int64, int64, int64, int64, int64), cache=True, fastmath=False, parallel=True)
def central_potential_J_coupling_matrix_element(cp_moshinsky_matrix, moshinsky_brackets, wigner_9j_dict, 
                                                n1, l1, twoj1, n2, l2, twoj2,
                                                n3, l3, twoj3, n4, l4, twoj4, twoJ):

    twos1, twos2, twos3, twos4 = 1, 1, 1, 1

    if (l1 != l3 or l2 != l4) and (l1 != l4 or l2 != l3):
        return 0.
    if (twoj1 != twoj3 or twoj2 != twoj4) and (twoj1 != twoj4 or twoj2 != twoj3):
        return 0.
    
    # Energy "numbers"
    Nq_12_i = 2 * n1 + l1 + 2 * n2 + l2
    Nq_34_i = 2 * n3 + l3 + 2 * n4 + l4

    Nq_12 = Nq_12_i #min(Nq_12_i, Nq_34_i)
    Nq_34 = Nq_34_i #max(Nq_12_i, Nq_34_i)

    # if Nq_12 > Nq_34:
    #     return 0.

    # Global multiplier
    glob_mult = np.sqrt(twoj1 + 1) * np.sqrt(twoj2 + 1) * np.sqrt(twoj3 + 1) * np.sqrt(twoj4 + 1)
    # For central potentials (with or without spin-spin interaction) we have deltas in l, N, L, lamb, S.
    # We use this to simplify the loop.
    mat_el = 0.
    coef_sum = 0.
    s1_counter = 0
    s0_counter = 0
    
    # Non-numba testing (run "export NUMBA_DISABLE_JIT='1'" in the terminal before running the script)
    quantum_numbers = []

    for n in prange(Nq_34//2 + 1):
        for l in prange(Nq_34 - 2*n + 1):
            for N in prange((Nq_34 - 2*n - l)//2 + 1):
                # L is fixed by conservation of energy: L = Nq_34 - 2n - l - 2N
                L = Nq_34 - 2*n - l - 2*N
                # if L < 0:
                #     raise ValueError("L < 0")

                for lamb in prange(abs(l - L), L + l + 1):
                    lamb_mult = 2 * lamb + 1

                    # Triangle inequality
                    if lamb < abs(l1 - l2) or lamb > l1 + l2:
                        continue
                    if lamb < abs(l3 - l4) or lamb > l3 + l4:
                        continue
                    
                    # n_prime is fixed by conservation of energy: 2*n_prime = Nq_12 - l - 2N - L
                    n_prime = (Nq_12 - l - 2*N - L) // 2
                    # n_prime must be positive for the conservation of energy to hold
                    if n_prime < 0:
                        continue
                    #n_prime = n + n3 - n1 + n4 - n2 - (l3 + l4 - l1 + l2)//2

                    # Actual candidate...:
                    moshinsky_term = moshinsky_brackets[n1, l1, n2, l2, n_prime, l, L, lamb] \
                                        * moshinsky_brackets[n3, l3, n4, l4, n, l, L, lamb] * (-1)**(L + l + lamb)
                                            
                    # moshinsky_term = moshinsky_brackets[n2, l2, n1, l1, n_prime, l, L, lamb] \
                    #                     * moshinsky_brackets[n4, l4, n3, l3, n, l, L, lamb] * (-1)**(L + l + lamb)

                    moshinsky_term = moshinsky_brackets[(N, L, n_prime, l, n1, l1, l2, lamb)] \
                                        * moshinsky_brackets[(N, L, n, l, n3, l3, l4, lamb)] * (-1)**(l1 + l2 + l3 + l4)
                    


                    quantum_numbers.append((n, l, N, L, lamb, 0, n_prime))
                    
                    if moshinsky_term == 0:
                        continue
                    
                    for twoS in range(0, 3, 2):
                        #print("n, l, N, L, lamb, twoS", n, l, N, L, lamb, twoS)
                        # Triangle inequality
                        if twoJ//2 < abs(twoS//2 - lamb) or twoJ//2 > (twoS//2 + lamb):
                            #print(twoS//2, lamb, twoJ//2)
                            continue

                        # TODO: changing this to 1 does not change the result. Why?
                        # It makes sense, as the triplet component should die.
                        # However, the singlet component should NOT die.
                        S_mult = (twoS + 1)

                        # if (2*l1, twos1, twoj1, 2*l2, twos2, twoj2, 2*lamb, twoS, twoJ) not in wigner_9j_dict:
                        #     print("abcdefg", (2*l1, twos1, twoj1, 2*l2, twos2, twoj2, 2*lamb, twoS, twoJ))

                        wigner_one = wigner_9j_dict[(2*l1, twos1, twoj1, 2*l2, twos2, twoj2, 2*lamb, twoS, twoJ)]
                        wigner_two = wigner_9j_dict[(2*l3, twos3, twoj3, 2*l4, twos4, twoj4, 2*lamb, twoS, twoJ)]
                        wigner_9j_term = wigner_one * wigner_two 

                        moshinsky_basis_el = cp_moshinsky_matrix[n_prime, l, N, L, lamb, twoS, n]
                        
                        # if twoS == 2:
                        #     #print("S", lamb_mult * S_mult * wigner_9j_term * moshinsky_term * moshinsky_basis_el)
                        #     mat_el += 0
                        #     s1_counter += 1
                        # else:
                        #     mat_el += lamb_mult * S_mult * wigner_9j_term * moshinsky_term * moshinsky_basis_el
                        #     s0_counter += 1
                        
                        mat_el += lamb_mult * S_mult * wigner_9j_term * moshinsky_term * moshinsky_basis_el

    # Check if repeated elements in quantum_numbers
    if len(quantum_numbers) != len(set(quantum_numbers)):
        print("HOL UPPPPP")
    
    quantum_numbers = np.array(quantum_numbers)
    #Check if columns --- contain the same elements
    # First, get these columns as lists
    col0 = quantum_numbers[:, 1].tolist()
    col2 = quantum_numbers[:, 3].tolist()
    # Then, sort them
    col0.sort()
    col2.sort()
    # Then, compare them
    if col0 != col2:
        print("2 HOL UPPPPP")
        print(col0)
        print(col2)


    mat_el *= glob_mult

    # if s1_counter == 0:
    #     print("s0", s0_counter, "s1", s1_counter, mat_el)

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
    full_brackets, _, _, _, _, _, _, _, brackets_dict = convert_fortran.get_moshinsky_arrays(saves_dir=SAVES_DIR)
    # Get the correct slice
    # n1, l1, n2, l2, n1', l1', l2', lamb   (  n2' = (2*n1 + l1 + 2*n2 + l2 - 2*n1' - l1' - l2')/2  )
    
    #print(full_brackets.shape)
    # The need for these large maxima comes from the calculation of the ls basis elements (this may be wrong)
    # You can probably go way smaller if you check the limits better
    ni_max = Ne_max
    li_max = 2 * Ne_max

    #moshinsky_brackets = full_brackets[0:ni_max+1, 0:li_max+1, 0:ni_max+1, 0:li_max+1, 0:ni_max+1, 0:li_max+1, 0:li_max+1, 0:2*li_max+1]
    moshinsky_brackets = full_brackets # FIXME

    return moshinsky_brackets


# load the Wigner 9j dictionary
def set_wigner_9js(numba=False):
    # Load the pickle file
    with open('{}/wigner_9js.pcl'.format(SAVES_DIR), 'rb') as handle:
        wigner_9j_dict = pickle.load(handle)
    
    if numba:
        wigner_9js_numba = Dict.empty(Tuple((int64, int64, int64, int64, int64, int64, int64, int64, int64)), float64)
        for key, value in wigner_9j_dict.items():
            wigner_9js_numba[key] = value
        
        print(type(wigner_9js_numba))
        return wigner_9js_numba

    return wigner_9j_dict


# This is Minnesota specific
# TODO: do this
@njit(float64[:,:,:,:](float64[:,:,:], float64, float64, int64, int64, float64, int64), parallel=True, fastmath=False, cache=True)
def set_central_potential_reduced_matrix(wavefunctions, V0, mu, Ne_max, l_max, integration_limit, integration_steps):

    Nq_lim = 2 * Ne_max # THIS IS NOT Nq_max!!! (the parameter in the function that calls this one, to fill the matrix)

    # These limits are overkill, but it's fine for now, this does not take much computing time
    central_potential_reduced_matrix = np.zeros((Nq_lim // 2 + 1, Nq_lim + 1, Nq_lim // 2 + 1, Nq_lim + 1), dtype=np.float64) 

    mat_el = 0
    for small_n in prange(Nq_lim // 2 + 1):
        for small_l in prange(Nq_lim + 1): # prange(Nq_lim - 2 * small_n + 1): --> This is the proper limit (the other is just for testing)
            for small_n_prime in prange(Nq_lim // 2 + 1):
                for small_l_prime in prange(Nq_lim + 1):    

                    central_potential_reduced_matrix[small_n_prime, small_l_prime, small_n, small_l] =\
                            central_potential_reduced_matrix_element(wavefunctions, V0, mu,
                                    small_n_prime, small_l_prime, small_n, small_l, integration_limit, integration_steps)   

    return central_potential_reduced_matrix


# Both for the CP and CP*spin-spin matrices
@njit(float64[:,:,:, :,:,:, :](float64[:,:,:,:], int64, int64, boolean), parallel=True, fastmath=False, cache=True)
def set_central_potential_moshinsky_basis_matrix(cp_red_matrix, Ne_max, l_max, spin_spin):
    ni_max = Ne_max
    li_max = 2 * Ne_max
    lambda_max = 2*li_max

    # n' l N L lamb S n (independent of J, M_J, and diagonal in l N L lamb S)
    central_potential_moshinsky_basis_matrix = np.zeros((ni_max+1, li_max+1, ni_max+1, li_max+1, lambda_max+1, 3, ni_max+1)) 
    
    # We can exploit the symmetry of the matrix elements to reduce the number of calculations (i.e., the deltas)
    # Deltas in l, N, L, lamb, S in both pure CP and CP*spin-spin
    # Set the central potential (in ls coupling basis) matrix
    for n in prange(ni_max + 1):
        for n_prime in prange(ni_max + 1):
            for l in prange(li_max + 1):
                for N in prange(ni_max + 1):
                    for L in prange(li_max + 1):
                        for lamb in prange(abs(l - L), L + l + 1):
                            for S in prange(2):
                            # TODO: n_prime will constrained by n and Ne_max in this case (not here, but later)
                                
                                if spin_spin:
                                    mat_el = cp_spin_spin_moshinsky_basis_matrix_element(cp_red_matrix, 
                                                    n_prime, l, N, L, lamb, 2*S,
                                                    n, l, N, L, lamb, 2*S, twoJ=0) # independent of J
                                else:
                                    mat_el = central_potential_moshinsky_basis_matrix_element(cp_red_matrix, 
                                                    n_prime, l, N, L, lamb, 2*S,
                                                    n, l, N, L, lamb, 2*S, twoJ=0)

                                central_potential_moshinsky_basis_matrix[n_prime, l, N, L, lamb, 2*S, n] =\
                                    mat_el

    return central_potential_moshinsky_basis_matrix


@njit(float64[:,:,:, :,:,:, :,:,:, :,:,:, :](float64[:,:,:,:, :,:,:], float64[:,:,:,:, :,:,:,:], DictType(UniTuple(int64, 9), float64), int64, int64),
        cache=True, parallel=True, fastmath=False)
def set_central_potential_J_coupling_basis_matrix(cp_moshinsky_matrix, moshinsky_brackets, wigner_9j_dict, Ne_max, l_max):
    ni_max = Ne_max // 2
    twoj_max = 2 * l_max + 1
    twoJ_max = 2 * twoj_max

    # TODO: Does this matrix need to be diagonal in l, l', j, j' for a general choice of potential?
    # Even for spin-spin? -> Yes, it does

    central_potential_J_coupling_basis_matrix = np.zeros((ni_max+1, l_max+1, 2, ni_max+1, l_max+1, 2, ni_max+1, l_max+1, 2, ni_max+1, l_max+1, 2, twoJ_max + 1))

    ns = []

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

                                        #ns.append((n1,n2,n3,n4))

                                        val = central_potential_J_coupling_matrix_element(cp_moshinsky_matrix, moshinsky_brackets,
                                                                                        wigner_9j_dict, n1, l, twoj, n2, l_prime,
                                                                            twoj_prime, n3, l, twoj, n4, l_prime, twoj_prime, twoJ)
                                        
                                        central_potential_J_coupling_basis_matrix[n1, l, twoj_idx, n2, l_prime, twoj_prime_idx,\
                                                                        n3, l, twoj_idx, n4, l_prime, twoj_prime_idx, twoJ] = val
                                        
                                        val = central_potential_J_coupling_matrix_element(cp_moshinsky_matrix, moshinsky_brackets,
                                                                                        wigner_9j_dict, n1, l, twoj, n2, l_prime,
                                                                            twoj_prime, n4, l_prime, twoj_prime, n3, l, twoj, twoJ)
                                        
                                        central_potential_J_coupling_basis_matrix[n1, l, twoj_idx, n2, l_prime, twoj_prime_idx,\
                                                                        n4, l_prime, twoj_prime_idx, n3, l, twoj_idx, twoJ] = val
                                        
                                
                                        # val = central_potential_J_coupling_matrix_element(cp_moshinsky_matrix, moshinsky_brackets,
                                        #                                                 wigner_9j_dict, n2, l_prime, twoj_prime, n1, l, twoj,
                                        #                                                  n4, l_prime, twoj_prime, n3, l, twoj, twoJ)
                                        
                                        # central_potential_J_coupling_basis_matrix[n2, l_prime, twoj_prime_idx, n1, l, twoj_idx, \
                                        #                                  n4, l_prime, twoj_prime_idx, n3, l, twoj_idx, twoJ] = val

                                        # val = central_potential_J_coupling_matrix_element(cp_moshinsky_matrix, moshinsky_brackets,
                                        #         wigner_9j_dict, n2, l_prime, twoj_prime, n1, l, twoj,
                                        #             n3, l, twoj, n4, l_prime, twoj_prime, twoJ)

                                        # central_potential_J_coupling_basis_matrix[n2, l_prime, twoj_prime_idx, n1, l, twoj_idx, \
                                        #                                   n3, l, twoj_idx, n4, l_prime, twoj_prime_idx, twoJ] = val

                                        # val = central_potential_J_coupling_matrix_element(cp_moshinsky_matrix, moshinsky_brackets,
                                        #                                                 wigner_9j_dict, n3, l, twoj, n4, l_prime, twoj_prime,
                                        #                                                   n1, l, twoj, n2, l_prime, twoj_prime, twoJ)
                                        
                                        # central_potential_J_coupling_basis_matrix[n3, l, twoj_idx, n4, l_prime, twoj_prime_idx,
                                        #                                           n1, l, twoj_idx, n2, l_prime, twoj_prime_idx, twoJ] = val
                                        

                                        # val = central_potential_J_coupling_matrix_element(cp_moshinsky_matrix, moshinsky_brackets,
                                        #                                                 wigner_9j_dict, n3, l, twoj, n4, l_prime, twoj_prime,
                                        #                                                   n2, l_prime, twoj_prime, n1, l, twoj, twoJ)
                                        
                                        # central_potential_J_coupling_basis_matrix[n3, l, twoj_idx, n4, l_prime, twoj_prime_idx,
                                        #                                           n2, l_prime, twoj_prime_idx, n1, l, twoj_idx, twoJ] = val
                                        
    # ns = np.array(ns)
    # # for i in range(ns.shape[0]):
    # #     print(ns[i, :])

    # col0 = ns[:, 0].tolist()
    # col2 = ns[:, 3].tolist()
    # # Then, sort them
    # col0.sort()
    # col2.sort()
    # # Then, compare them
    # if col0 != col2:
    #     print("2 HOL UPPPPP")
    #     print(col0)
    #     print(col2)

    # exit()

    return central_potential_J_coupling_basis_matrix


@njit(float64[:,:,:,:,:,:,:,:,:,:,:,:](float64[:,:,:,:,:,:,:,:,:,:,:,:,:], int64, int64), parallel=True, fastmath=False, cache=True)
def set_central_potential_matrix(cp_J_coupling_matrix, Ne_max, l_max):
    ni_max = Ne_max // 2
    
    central_potential_basis_matrix = np.zeros((ni_max+1, l_max+1, 2, ni_max+1, l_max+1, 2, ni_max+1, l_max+1, 2, ni_max+1, l_max+1, 2))

    added_indices = []
    added_indices_swap = []
    twoJs = np.arange(0, 4*l_max + 3, 1, dtype=np.int64)

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

                                    central_potential_basis_matrix[n1, l, twoj_idx, n2, l_prime, twoj_prime_idx, n3, l, twoj_idx, n4, l_prime, twoj_prime_idx] =\
                                    np.sum(cp_J_coupling_matrix[n1, l, twoj_idx, n2, l_prime, twoj_prime_idx, n3, l, twoj_idx, n4, l_prime, twoj_prime_idx, :] *\
                                            (twoJs + 1)) * 1 / ((twoj + 1) * (twoj_prime + 1))

                                    central_potential_basis_matrix[n1, l, twoj_idx, n2, l_prime, twoj_prime_idx, n4, l_prime, twoj_prime_idx, n3, l, twoj_idx] =\
                                    np.sum(cp_J_coupling_matrix[n1, l, twoj_idx, n2, l_prime, twoj_prime_idx, n4, l_prime, twoj_prime_idx, n3, l, twoj_idx, :] *\
                                        (twoJs + 1)) * 1 / ((twoj + 1) * (twoj_prime + 1))
                                    
                                    # central_potential_basis_matrix[n2, l_prime, twoj_prime_idx, n1, l, twoj_idx, n4, l_prime, twoj_prime_idx, n3, l, twoj_idx] =\
                                    # np.sum(cp_J_coupling_matrix[n2, l_prime, twoj_prime_idx, n1, l, twoj_idx, n4, l_prime, twoj_prime_idx, n3, l, twoj_idx, :] *\
                                    #     (twoJs + 1)) * 1 / ((twoj + 1) * (twoj_prime + 1))
                                    
                                    # central_potential_basis_matrix[n2, l_prime, twoj_prime_idx, n1, l, twoj_idx, n3, l, twoj_idx, n4, l_prime, twoj_prime_idx] =\
                                    # np.sum(cp_J_coupling_matrix[n2, l_prime, twoj_prime_idx, n1, l, twoj_idx,n3, l, twoj_idx, n4, l_prime, twoj_prime_idx, :] *\
                                    #     (twoJs + 1)) * 1 / ((twoj + 1) * (twoj_prime + 1))

                                    # central_potential_basis_matrix[n3, l, twoj_idx, n4, l_prime, twoj_prime_idx, n1, l, twoj_idx, n2, l_prime, twoj_prime_idx] =\
                                    # np.sum(cp_J_coupling_matrix[n3, l, twoj_idx, n4, l_prime, twoj_prime_idx, n1, l, twoj_idx, n2, l_prime, twoj_prime_idx, :] *\
                                    #         (twoJs + 1)) * 1 / ((twoj + 1) * (twoj_prime + 1))
                                    
                                    # central_potential_basis_matrix[n3, l, twoj_idx, n4, l_prime, twoj_prime_idx, n2, l_prime, twoj_prime_idx, n1, l, twoj_idx] =\
                                    # np.sum(cp_J_coupling_matrix[n3, l, twoj_idx, n4, l_prime, twoj_prime_idx, n2, l_prime, twoj_prime_idx, n1, l, twoj_idx, :] *\
                                    #         (twoJs + 1)) * 1 / ((twoj + 1) * (twoj_prime + 1))


    

    return central_potential_basis_matrix

# This is horrible python code, I'm just trying to make it work with numba.
# God, please forgive me.
def main(Ne_max, l_max, hbar_omega, mass, integration_limit, integration_steps):

    # Set the radial wavefunctions
    wfs = set_wavefunctions(Ne_max, l_max, hbar_omega, mass, integration_limit, integration_steps)

    r = np.linspace(0, integration_limit, integration_steps)

    # plt.plot(r, wfs[0,0,:])
    # plt.plot(r, wfs[1,0,:])
    # plt.plot(r, wfs[1,1,:])
    plt.plot(r, wfs[2,2,:])
    plt.show()
    
    V0 = 200
    kappa = 1.487

    central_potential_reduced_matrix = set_central_potential_reduced_matrix(wfs, V0, kappa,
                                                        Ne_max, l_max, integration_limit, integration_steps)

    #print(central_potential_reduced_matrix[0,0,:,:])

    print(central_potential_reduced_matrix_element(wfs, V0, kappa, 1, 2, 1, 2, integration_limit, integration_steps))
    print(central_potential_reduced_matrix[1,2,1,2])

    


if __name__ == '__main__':
    # Call the function
    
    main(Ne_max=6, l_max=2, hbar_omega=10, mass=939, integration_limit=40, integration_steps=5500)

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
