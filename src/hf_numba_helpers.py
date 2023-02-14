import numpy as np
from numba import jit, njit, prange
from numba.types import intc, CPointer, float64, int32, int64, boolean


# Fast change of basis for the 4D potential matrix
@njit(float64[:,:,:,:](float64[:,:,:,:], float64[:,:]), parallel=True, fastmath=True, cache=True)
def n_change_basis_4d(matrix, U):
    U_dagger = U.conj().T

    num_states = matrix.shape[0]
    new_matrix = np.zeros((num_states, num_states, num_states, num_states))
    for i in prange(num_states):
        for j in prange(num_states):
            print(i, j)
            for k in prange(num_states):
                for l in prange(num_states):
                    for m in prange(num_states):
                        for n in prange(num_states):
                            for o in prange(num_states):
                                for p in prange(num_states):
                                    new_matrix[i, j, k, l] += U[i, m] * U[j, n] * U[k, o] * U[l, p] * matrix[m, n, o, p] * U_dagger[m, i] * U_dagger[n, j] * U_dagger[o, k] * U_dagger[p, l]
    
    return new_matrix
