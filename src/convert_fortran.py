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

    if type (Nq) is not int:
        if ((Nq_max - Nq) % 2).any() != 0:
            raise ValueError("Nq_max - Nq is not even")

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

    n2prime = (-2 * n1prime - l1prime - l2prime + 2*n1 + 2*n2 + l1 + l2)//2
    if not np.allclose(2 * Aprime + 2 * n1prime + 2 * n2prime + lamb + eps, Nq):
        raise ValueError("2 * Aprime + 2 * n1prime + 2 * n2prime != Nq")

    if type(Bprime) is not int:
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
        #return [n1, l1, n2, l2, n1prime, l1prime, l2prime, lamb]
        # Swap the first two indices to get "normal" Moshinsky brackets (instead of generalized)
        return [n1prime, l1prime, n2prime, l2prime, n1, l1, l2, lamb]
        #return [n2prime, l2prime, n1prime, l1prime, n1, l1, l2, lamb]
    
    elif mode == 'Ne':
        Ne1 = 2 * n1 + l1
        Ne2 = 2 * n2 + l2
        Ne1prime = 2 * n1prime + l1prime
        return [Ne1, l1, Ne2, l2, Ne1prime, l1prime, l2prime, lamb]


# "complex conjugate" of the Moshinsky brackets, which are real
# n1, l1, n2, l2, n1', l1', l2', lamb --> n1', l1', n2', l2', n1, l1, l2, lamb
def get_swapped_indices(indices):
    #n1, l1, n2, l2, n1p, l1p, l2p, lamb = indices[:,0], indices[:,1], indices[:,2], indices[:,3], indices[:,4], indices[:,5], indices[:,6], indices[:,7]
    n1, l1, n2, l2, n1p, l1p, l2p, lamb = indices.T

    n2p = (2*n1 + l1 + 2*n2 + l2 - 2*n1p - l1p - l2p)//2

    return np.vstack((n1p, l1p, n2p, l2p, n1, l1, l2, lamb)).T


def get_multipliers(indices):
    #n1p, l1p, n2p, l2p, n1, l1, l2, lamb = indices.T
    n2p, l2p, n1p, l1p, n1, l1, l2, lamb = indices.T
    n2 = (2*n1p + l1p + 2*n2p + l2 - 2*n1 - l1 - l2)//2

    mult = 1#(-1) ** (l1p + lamb)

    return mult


def get_moshinsky_arrays(saves_dir=None):
    # Prepare the field
    lamb_max = 16
    lamb_min = 0
    #nqdif = 2
    Nq_max_even = 20
    Nq_max_odd = 21
    Nq_max = max(Nq_max_even, Nq_max_odd)

    sz = (Nq_max - lamb_min)//2 + 1
    brackets_f = np.zeros((lamb_max + 1, sz, sz, sz, sz, lamb_max + 1, sz, lamb_max - lamb_min + 1))
    brackets_f_even = np.zeros((lamb_max + 1, sz, sz, sz, sz, lamb_max + 1, sz, lamb_max - lamb_min + 1))
    brackets_f_odd = np.zeros((lamb_max + 1, sz, sz, sz, sz, lamb_max + 1, sz, lamb_max - lamb_min + 1))

    # Open the file and load the Fortran computed values
    values_and_indices_f_even = np.loadtxt('{}/out_even.dat'.format(saves_dir))
    values_and_indices_f_odd = np.loadtxt('{}/out_odd.dat'.format(saves_dir))
    values_f_even = values_and_indices_f_even[:, -1]
    values_f_odd = values_and_indices_f_odd[:, -1]
    indices_f_even = values_and_indices_f_even[:, :-1].astype(int)
    indices_f_odd = values_and_indices_f_odd[:, :-1].astype(int)
    values_f = np.hstack((values_and_indices_f_even[:, -1], values_and_indices_f_odd[:, -1]))
    indices_f = np.vstack((values_and_indices_f_even[:, :-1].astype(int), values_and_indices_f_odd[:, :-1].astype(int)))

    # Index the matrix
    # Based on the stupid order things are saved in FORTRAN (yes, makes no sense)
    # N' (B'), n1', M' (A'), n1, n2, N (B), M (A), L (lamb)
    brackets_f_odd[tuple(indices_f_odd.T)] = values_f_odd
    brackets_f_even[tuple(indices_f_even.T)] = values_f_even

    # tn = 17
    # print(indices_f[tn,:])
    # print(values_f[tn])
    # print(brackets_f[indices_f[tn,0], indices_f[tn,1], indices_f[tn,2], indices_f[tn,3], indices_f[tn,4], indices_f[tn,5], indices_f[tn,6], indices_f[tn,7]])
    
    # print(brackets_f_even[0,0, 0, 0, 0,0, 1, 1])
    # print(indices_f_even[12636,:])


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

    # print(recast_indices('n', Nq_max_even, 0, 0, 0, 0, 0, 1, 1, 2))
    # print(recast_indices('n', Nq_max_even, 1, 0, 1, 0, 1, 0, 0, 2))

    # print(indices_even[13,:])
    # print(indices_f_even[13,:], "HHJHJH")
    # exit()

    # We also want to add the "swapped" coefficients, which should be the same
    # (n1, l1, n2, l2, n1', l1', l2', lamb) -> (n1', l1', n2', l2', n1, l1, l2, lamb)
    swapped_indices_even = get_swapped_indices(indices_even)
    swapped_indices_odd = get_swapped_indices(indices_odd)
    # print(swapped_indices_even.shape, indices_even.shape)
 
    #Check for collisions (empty!)
    # print(indices_even.shape)
    # print(indices_odd.shape)
    # eset = set([tuple(row) for row in indices_even])
    # oset = set([tuple(row) for row in indices_odd])
    # inter = set.intersection(eset, oset) #np.array([x for x in eset & oset])
    # print(inter)
    #exit()

    # seset = set([tuple(row) for row in swapped_indices_even])
    # inter_es = set.intersection(eset, seset)
    #print(inter_es)

    # Get multipliers
    multipliers_even = get_multipliers(indices_even)
    multipliers_odd = get_multipliers(indices_odd)

    # Index the matrix
    brackets[tuple(indices_even.T)] = values_f_even * multipliers_even
    brackets[tuple(indices_odd.T)] = values_f_odd * multipliers_odd

    # Create dictionary
    # Create a dictionary from the nonzero values
    brackets_dict = {}
    for i in range(len(indices_even)):
        brackets_dict[tuple(indices_even[i])] = values_f_even[i]
    for i in range(len(indices_odd)):
        brackets_dict[tuple(indices_odd[i])] = values_f_odd[i]


    #brackets[tuple(swapped_indices_even.T)] = values_f_even
    # brackets[tuple(swapped_indices_odd.T)] = values_f_odd

    # for val in inter_es:
    #     n1, l1, n2, l2, n1p, l1p, l2p, lamb = val 
    #     n2p = (2*n1 + l1 + 2*n2 + l2 - 2*n1p - l1p - l2p)//2

    #     print(val, brackets[tuple(val)])
    #     print(val, brackets[n1p, l1p, n2p, l2p, n1, l1, l2, lamb])


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

    return brackets, brackets_Ne, indices_even, indices_odd, brackets_f_even, indices_f_even, brackets_f_odd, indices_f_odd, brackets_dict