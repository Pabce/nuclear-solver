import numpy as np
from findiff import FinDiff
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from scipy.linalg import eigh

class S_solver:

    def __init__(self, potential):
        self.potential = potential
    

    def solve(self, r, potential_parameters, mass, number_of_states=10):

        # Solve the radial Schrödinger equation
        # Create the Hamiltonian
        # Laplacian operator:
        lap = FinDiff(0, r, 2).matrix(r.shape)

        # Hamiltonian
        hamiltonian = -lap / (2 * mass) + np.diag(self.potential(r, *potential_parameters))
        eigenvalues, eigenvectors = eigh(hamiltonian, subset_by_index=[0, number_of_states])

        # hamiltonian = -lap / (2 * red_mass) + diags(effective_potential(r, j, l, A, Z, nucleon, r0, a, V0, r0_so, lamb, kappa))
        # eigenvalues, eigenvectors = eigs(hamiltonian, k=10, which='SR') # Seems slower!

        return eigenvalues, eigenvectors