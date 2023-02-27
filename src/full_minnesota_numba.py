import numpy as np
from numba import cfunc, jit, njit, prange
from numba.types import intc, CPointer, float64, int32, int64, boolean


# --------------------------------------------------------------------------------------------------------
# For the Minnesota system (probably will be reused for other systems as well)
# Numba methods to be used by the solver class to convert between quantum numbers and flat indices

# twom == 0 indicates we are not including the m quantum number in the index
@njit(int64(int64, int64, int64, int64, int64, int64), fastmath=True, cache=True)
def n_index_flattener(n, l, twoj, twom, Ne_max, l_level_max):
    
    if twoj > 2 * l + 1 or twoj < np.abs(2 * l - 1):
        print("The value of twoj is not allowed for the given value of l", twoj, l)
    
    if twom != 0:
        if np.abs(twom) > twoj or (twoj - twom) % 2 != 0:
            print("The value of twom is not allowed for the given value of twoj", twom, twoj)

    idx = 0
    for n_p in prange(n + 1):
        Nep_min = 2 * n_p
        lp_max = min(l_level_max, Ne_max - Nep_min)

        if n_p == n and l > lp_max:
            print("The value of l is too large for the given value of n (for the system's Ne_max, l_level_max)",
                                                                                                    l, n, Ne_max, l_level_max)

        for lp in prange(lp_max + 1):
            for twojp in prange(np.abs(2 * lp - 1), 2 * lp + 2, 2):
                
                if not twom:
                    if n_p == n and lp == l and twojp == twoj:
                        return idx
                    idx += 1
                else:
                    for twomp in prange(-twojp, twojp + 1, 2):
                        if n_p == n and lp == l and twojp == twoj and twomp == twom:
                            return idx
                        idx += 1
    
    print("Something went wrong")
    return -1


@njit(float64[:,:,:,:](float64[:,:,:,:,:,:,:,:,:,:,:,:], boolean, boolean, boolean, int64, int64, int64), parallel=True, cache=True)
def n_matrix_4dflatten(matrix, include_m, m_diagonal, asym, num_states, Ne_max, l_level_max):
    dim = 4
    flat_matrix = np.zeros((num_states, num_states, num_states, num_states))

    idx = np.zeros(dim, dtype=int64)

    for n1 in prange(Ne_max//2 + 1):
        for l1 in prange(l_level_max + 1):
            for twoj1_idx in prange(2):
                for n2 in prange(Ne_max//2 + 1):
                    for l2 in prange(l_level_max + 1):
                        for twoj2_idx in prange(2):
                            for n3 in prange(Ne_max//2 + 1):
                                for l3 in prange(l_level_max + 1):
                                    for twoj3_idx in prange(2):
                                        for n4 in prange(Ne_max//2 + 1):
                                            for l4 in prange(l_level_max + 1):
                                                for twoj4_idx in prange(2):
                                                    el = matrix[n1, l1, twoj1_idx, n2, l2, twoj2_idx, n3, l3, twoj3_idx, n4, l4, twoj4_idx]
                                                    twoj1 = 2 * l1 - 1 + twoj1_idx * 2
                                                    twoj2 = 2 * l2 - 1 + twoj2_idx * 2
                                                    twoj3 = 2 * l3 - 1 + twoj3_idx * 2
                                                    twoj4 = 2 * l4 - 1 + twoj4_idx * 2

                                                    if twoj1 < 0 or twoj2 < 0 or twoj3 < 0 or twoj4 < 0:
                                                        continue
                                                    if 2 * n1 + l1 > Ne_max or 2 * n2 + l2 > Ne_max or 2 * n3 + l3 > Ne_max or 2 * n4 + l4 > Ne_max:
                                                        continue
                                                    
                                                    if include_m:

                                                        for twom_1 in range(-twoj1, twoj1 + 1, 2):
                                                            for twom_2 in range(-twoj2, twoj2 + 1, 2):
                                                                for twom_3 in range(-twoj3, twoj3 + 1, 2):
                                                                    for twom_4 in range(-twoj4, twoj4 + 1, 2):

                                                                        idx1 = n_index_flattener(n1, l1, twoj1, twom_1, Ne_max, l_level_max)
                                                                        idx2 = n_index_flattener(n2, l2, twoj2, twom_2, Ne_max, l_level_max)
                                                                        idx3 = n_index_flattener(n3, l3, twoj3, twom_3, Ne_max, l_level_max)
                                                                        idx4 = n_index_flattener(n4, l4, twoj4, twom_4, Ne_max, l_level_max)

                                                                        non_zero = True
                                                                        if m_diagonal:
                                                                            # TODO: this should be changed if the matrix you are flattening is not antysymmetrized in the last two indices
                                                                            non_zero = False
                                                                            if asym:
                                                                                if (twom_1 == twom_3 and twom_2 == twom_4):
                                                                                    # print(twom, m1, m2)
                                                                                    flat_matrix[idx1, idx2, idx3, idx4] = el
                                                                                elif (twom_1 == twom_4 and twom_2 == twom_3):
                                                                                    flat_matrix[idx1, idx2, idx3, idx4] = el
                                                                            else:
                                                                                if (twom_1 == twom_3 and twom_2 == twom_4):
                                                                                    flat_matrix[idx1, idx2, idx3, idx4] = el

                                                                        else:
                                                                            flat_matrix[idx1, idx2, idx3, idx4] = el
                                                    else:
                                                        idx1 = n_index_flattener(n1, l1, twoj1, 0, Ne_max, l_level_max)
                                                        idx2 = n_index_flattener(n2, l2, twoj2, 0, Ne_max, l_level_max)
                                                        idx3 = n_index_flattener(n3, l3, twoj3, 0, Ne_max, l_level_max)
                                                        idx4 = n_index_flattener(n4, l4, twoj4, 0, Ne_max, l_level_max)

                                                        flat_matrix[idx1, idx2, idx3, idx4] = el
    
    return flat_matrix