import numpy as np
from numba import jit, njit
from scipy.sparse import bsr_array


def get_nq_and_eps(Nq_max, A, lamb, n1, n2):
    # Determine the parity of Nq - lamb
    Nqpar = Nq_max % 2
    Nq_minus_lamb_par = (Nqpar - lamb) % 2
    eps = Nq_minus_lamb_par % 2
    # Determine Nq
    Nq = 2 * n1 + 2 * n2 + 2 * A + lamb + eps

    return Nq, eps


def get_ells(A, B, lamb, eps):    
    l1 = A + B + eps
    l2 = A - B + lamb

    return l1, l2

# Based on the stupid order things are saved in FORTRAN (yes, makes no sense)
# From: N' (B'), n1', M' (A'), n1, n2, N (B), M (A), L (lamb)
# To: n1, l1, n2, l2, n1', l1', l2', lamb  (  n2' = (2*n1 + l1 + 2*n2 + l2 - 2*n1' - l1' - l2')/2  )
# Or: Ne1, l1, Ne2, l2, Ne1', l1', l2', lamb  (  Ne2' = Ne1 + Ne2 - Ne1'  )
def recast_indices(mode, Nq_max, Bprime, n1prime, Aprime, n1, n2, B, A, lamb):
    Nq, eps = get_nq_and_eps(Nq_max, A, lamb, n1, n2)
    l1, l2 = get_ells(A, B, lamb, eps)
    l1prime, l2prime = get_ells(Aprime, Bprime, lamb, eps)

    for i in range(len(l1)):
        if (l1[i] + l2[i] - Nq[i]) % 2 != 0:
            print("Error in recast_indices: l1 + l2 - Nq is not even")
            print(l1[i], l2[i], Nq[i])
        
        if (l1prime[i] + l2prime[i] - Nq[i]) % 2 != 0:
            print("Error in recast_indices: l1prime + l2prime - Nq is not even")
            print(l1prime[i], l2prime[i], Nq[i])
        
        if 2 * n1[i] + l1[i] + 2 * n2[i] + l2[i] != Nq[i]:
            print("Error in recast_indices: n1 + l1 + n2 + l2 != Nq")
            print(n1[i], l1[i], n2[i], l2[i], Nq[i])

    if mode == 'n':
        return [n1, l1, n2, l2, n1prime, l1prime, l2prime, lamb]
    
    elif mode == 'Ne':
        Ne1 = 2 * n1 + l1
        Ne2 = 2 * n2 + l2
        Ne1prime = 2 * n1prime + l1prime

        return [Ne1, l1, Ne2, l2, Ne1prime, l1prime, l2prime, lamb]


def get_moshinsky_arrays():
    # Prepare the field
    lamb_max = 16
    lamb_min = 0
    #nqdif = 2
    Nq_max_even = 20
    Nq_max_odd = 21
    Nq_max = max(Nq_max_even, Nq_max_odd)

    sz = (Nq_max - lamb_min)//2 + 1
    brackets_f = np.zeros((lamb_max + 1, sz, sz, sz, sz, lamb_max + 1, sz, lamb_max - lamb_min + 1))

    # Open the file and load the Fortran computed values
    values_and_indices_f_even = np.loadtxt('OSBRACKETS/out_even.dat')
    values_and_indices_f_odd = np.loadtxt('OSBRACKETS/out_odd.dat')
    values_f_even = values_and_indices_f_even[:, -1]
    values_f_odd = values_and_indices_f_odd[:, -1]
    indices_f_even = values_and_indices_f_even[:, :-1].astype(int)
    indices_f_odd = values_and_indices_f_odd[:, :-1].astype(int)
    values_f = np.hstack((values_and_indices_f_even[:, -1], values_and_indices_f_odd[:, -1]))
    indices_f = np.vstack((values_and_indices_f_even[:, :-1].astype(int), values_and_indices_f_odd[:, :-1].astype(int)))

    # Index the matrix
    # Based on the stupid order things are saved in FORTRAN (yes, makes no sense)
    # N' (B'), n1', M' (A'), n1, n2, N (B), M (A), L (lamb)
    brackets_f[tuple(indices_f.T)] = values_f

    # --------------------------------------------
    # Brackets with the quantum numbers we want:
    # n1, l1, n2, l2, n1', l1', l2', lamb   (  n2' = (2*n1 + l1 + 2*n2 + l2 - 2*n1' - l1' - l2')/2  )
    l_min = 0
    l_max = Nq_max #lamb_max (there is an interesting reason why this is not lamb_max)
    n_max = (Nq_max - lamb_min)//2
    brackets = np.zeros((n_max + 1, l_max - l_min + 1, n_max + 1, l_max - l_min + 1, n_max + 1,
                                    l_max - l_min + 1, l_max - l_min + 1, lamb_max - lamb_min + 1))
    # Recast the indices (incredibly works)
    indices_even = np.vstack(recast_indices('n', Nq_max_even, *[indices_f_even[:, i] for i in range(8)])).T
    indices_odd = np.vstack(recast_indices('n', Nq_max_odd, *[indices_f_odd[:, i] for i in range(8)])).T
    # Index the matrix
    brackets[tuple(indices_even.T)] = values_f_even
    brackets[tuple(indices_odd.T)] = values_f_odd

    # print(indices_even.shape)
    # print(indices_odd.shape)

    # Some tests
    # print(indices_f[12])
    # print(recast_indices('n', Nq_max, *indices_f[12]))
    # print(indices[12])
    # print(values_f[12], brackets[tuple(indices[12].T)])

    # --------------------------------------------
    # Or:
    # Ne1, l1, Ne2, l2, Ne1', l1', l2', lamb  (  Ne2' = Ne1 + Ne2 - Ne1'  )
    # Ne_max = Nq_max
    # brackets_Ne = np.zeros((Ne_max + 1, l_max - l_min + 1, Ne_max + 1, l_max - l_min + 1, Ne_max + 1,
    #                                 l_max - l_min + 1, l_max - l_min + 1, lamb_max - lamb_min + 1))
    # # Recast the indices
    # indices_Ne = np.vstack(recast_indices('Ne', Nq_max, *[indices_f[:, i] for i in range(8)])).T
    # # Index the matrix
    # brackets_Ne[tuple(indices_Ne.T)] = values_f
    brackets_Ne = None

    return brackets, brackets_Ne