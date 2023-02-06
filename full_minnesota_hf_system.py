'''
Attempt to implement Hartree-Fock as shown in https://wikihost.nscl.msu.edu/TalentDFT/doku.php?id=projects
We start with a simplified neutron drop system in the harmonic oscillator basis.
'''

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.integrate import quad, dblquad
import harmonic_3d as h3d
import time
import ctypes
from scipy import LowLevelCallable
from numba import cfunc, jit
from numba.types import intc, CPointer, float64, int32, int64
from minnesota_cfuncs import c_potential_l0, nb_potential_l0
import moshinsky_way as mw
from itertools import product

HBAR = 1

# Neutron drop system
# For purposes of computing the matrix elements, and also because
# we are interested in chosing the lowest energy states, we will
# work with the quantum number N = 2n + l, calling it Ne here
# (as it represents the energy of the system)
# I hope I don't regret this
# Spoiler alert: I do regret this
class System:
    # Minnesota potential parameters
    V0R = 200 # MeV
    V0t = 178 # MeV
    V0s = 91.85 # MeV
    kappaR = 1.487 # fm^-2
    kappat = 0.639 # fm^-2
    kappas = 0.465 # fm^-2 

    def __init__(self, Ne_max, l_max, omega=1, mass=1) -> None:
        self.Ne_max = Ne_max
        # l_max represents the maximum value of l (cannot be larger than Ne_max)
        if l_max > Ne_max:
            raise ValueError("l_max cannot be larger than Ne_max")

        # l can go from Ne to 0 in steps of 2
        num_states = 0
        for Ne in range(Ne_max + 1):
            max_l = min(Ne, l_max)
            for l in range(Ne, -1, -2):
                if l > max_l:
                    continue
                for twoj in range(np.abs(2*l - 1), 2*l + 2, 2):
                    num_states += twoj + 1 # for the j and m_j degeneracy

        self.num_states = int(num_states)
        self.n_level_max = Ne_max // 2
        self.l_level_max = l_max

        self.omega = omega
        self.mass = mass

        # List of all energies and their indices
        self.eigenenergies, self.eigenenergies_flat, self.quantum_numbers = self.get_lowest_energies()

        self.wavefunctions, self.sqrt_norms = self.generate_wavefunctions()

        # The way the indices will work is idx = k + l * k_num + spin * k_num * l_max
        # (where here spin is 0 or 1)
    

    def generate_wavefunctions(self, r_limit = 15, r_steps = 2500):
        wavefunctions = np.zeros((self.Ne_max, self.l_level_max), dtype=object)
        sqrt_norms = np.zeros((self.Ne_max, self.l_level_max), dtype=np.float64)
        for Ne in range(self.Ne_max):
            for l in range(self.l_level_max):
                n = (Ne - l) / 2

                r = np.linspace(0, r_limit, r_steps)
                _, sqrt_norm = h3d.series_wavefunction(r, k=n, l=l, omega=self.omega, mass=self.mass)

                wavefunctions[Ne, l] = lambda r, k=n, l=l: h3d.wavefunction(r, k=k, l=l, omega=self.omega, mass=self.mass)
                sqrt_norms[Ne, l] = sqrt_norm

        return wavefunctions, sqrt_norms
    
    # r1 and r2 are vectors
    def pot_R(self, r1, r2):
        r12 = np.linalg.norm(r1 - r2)
        return self.V0R * np.exp(-self.kappaR * (r12)**2)
    
    def pot_t(self, r1, r2):
        r12 = np.linalg.norm(r1 - r2)
        return -self.V0t * np.exp(-self.kappat * (r12)**2)
    
    def pot_s(self, r1, r2):
        r12 = np.linalg.norm(r1 - r2)
        return -self.V0s * np.exp(-self.kappas * (r12)**2)

    
    # Get the one-body matrix elements in the HO basis
    def get_one_body_matrix_elements(self):
        # As given in https://wikihost.nscl.msu.edu/TalentDFT/lib/exe/fetch.php?media=ho_spherical.pdf,
        # the one-body matrix elements for this model in the HO basis are...
        # These elements only depend on n and l, not on the spin:

        t = np.zeros((self.Ne_max // 2 + 1, self.l_level_max + 1, 2, self.Ne_max // 2 + 1, self.l_level_max + 1, 2), dtype=np.float64)

        # They are also diagonal in l
        for n1, n2 in product(range(self.Ne_max // 2 + 1), repeat=2):
            for l in range(self.l_level_max + 1):
                if n1 == n2:
                    val = 2 * n1 + l + 3/2
                elif n1 == n2 - 1:
                    val = np.sqrt(n2 * (n2 + l + 1/2))
                elif n1 == n2 + 1:
                    val = np.sqrt(n1 * (n1 + l + 1/2))
                else:
                    val = 0
        
                t[n1, l, :, n2, l, :] = val

        t *= 0.5 * HBAR * self.omega
        
        return t
    

    # Get the antisymmetrized two-body matrix elements in the HO basis
    # TODO: reduce computations exploiting hermiticity of the potential
    def get_two_body_matrix_elements(self):

        Ne_max, l_max, omega, mass = self.Ne_max, self.l_level_max, self.omega, self.mass
        integration_limit, integration_steps = 10, 1000

        # TODO: Add a way to pass the potential into the Moshinsky class
        # Set the radial wavefunctions
        wfs = mw.set_wavefunctions(Ne_max, l_max, omega, mass, integration_limit, integration_steps)

        # Set the Moshinsky brackets, reading from file
        start = time.time()
        moshinsky_brackets = mw.set_moshinsky_brackets(Ne_max, l_max)
        print("Time to set Moshinsky brackets:", time.time() - start)

        # Set the central potential reduced matrix elements. You gave to do this for two values...
        V_mats = []
        for params in [(self.V0R, self.kappaR), (-self.V0s, self.kappas)]: 
            V0, kappa = params

            start = time.time()
            central_potential_reduced_matrix = mw.set_central_potential_reduced_matrix(wfs, V0, kappa,
                                                        Ne_max, l_max, integration_limit, integration_steps)
            print("Time to set reduced matrix elements:", time.time() - start)

            # Set the central potential matrix in ls coupling basis
            start = time.time()
            central_potential_ls_coupling_basis_matrix = mw.set_central_potential_ls_coupling_basis_matrix(central_potential_reduced_matrix,
                                                                                                        moshinsky_brackets, Ne_max, l_max)
            print("Time to set ls coupling basis matrix:", time.time() - start)

            # Set the central potential matrix in J coupling basis
            start = time.time()
            central_potential_J_coupling_matrix = mw.set_central_potential_J_coupling_basis_matrix(central_potential_ls_coupling_basis_matrix, Ne_max, l_max)
            print("Time to set J coupling basis matrix:", time.time() - start)
            
            # Get the central potential matrix, once and for all
            start = time.time()
            central_potential_matrix = mw.set_central_potential_matrix(central_potential_J_coupling_matrix, Ne_max, l_max)
            print("Time to set central potential matrix:", time.time() - start)

            V_mats.append(central_potential_matrix)

            print("------------------------------------------------------------------------------")

        vr, vs = V_mats
        vr_a, vs_a = V_mats

        VD = 0.5 * (vr + vs)
        VEPr = 0.5 * (vr_a + vs_a)

        V_mat = VD + VEPr

        return V_mat


    # Energies of the HO basis eigenstates
    def get_ho_energies(self, Ne=-1, n=-1, l=-1):
        if Ne == -1:
            return 0.5 * self.omega * (2 * n + l + 3/2)
        else:
            return 0.5 * self.omega * Ne

    # Get the indices of the lowest energy states
    def get_lowest_energies(self):
        energies_nl = np.zeros((self.n_level_max + 1, self.l_level_max + 1))
        energies_flat = []
        quantum_numbers = []

        for n in range(self.n_level_max + 1):
            for l in range(self.l_level_max + 1):
                e = self.get_ho_energies(n=n, l=l)
                energies_nl[n, l] = e

                for twoj in range(np.abs(2 * l - 1), 2 * l + 2, 2):
                    energies_flat.extend([e] * (twoj + 1))
                    quantum_numbers.extend([(n, l, twoj)] * (twoj + 1))

        #idx = np.argsort(energies)[:num]
        return energies_nl, energies_flat, quantum_numbers
    
    # --------------------------------------------------------------------------------------------------------
    # Methods to be used by the solver class to convert between quantum numbers and flat indices
    def index_flattener(self, n, l, twoj, twom):
        
        if twoj > 2 * l + 1 or twoj < np.abs(2 * l - 1):
            raise ValueError("The value of twoj={} is not allowed for the given value of l={}".format(twoj, l))
        if np.abs(twom) > twoj or (twoj - twom) % 2 != 0:
            raise ValueError("The value of twom={} is not allowed for the given value of twoj={}".format(twom, twoj))

        idx = 0
        for n_p in range(n + 1):
            Nep_min = 2 * n_p
            lp_max = np.min((self.l_level_max, self.Ne_max - Nep_min))

            if n_p == n and l > lp_max:
                raise ValueError("The value of l={} is too large for the given value of n={} (for the system's Ne_max={}, l_level_max={})".\
                                                                                                            format(l, n, self.Ne_max, self.l_level_max))

            for lp in range(lp_max + 1):
                for twojp in range(np.abs(2 * lp - 1), 2 * lp + 2, 2):
                    for twomp in range(-twojp, twojp + 1, 2):
                        if n_p == n and lp == l and twojp == twoj and twomp == twom:
                            return idx
                        idx += 1
        
        raise ValueError("Something went wrong")

    
    def index_unflattener(self, idx):

        idxp = 0
        for n_p in range(self.n_level_max + 1):
            Nep_min = 2 * n_p
            lp_max = np.min((self.l_level_max, self.Ne_max - Nep_min))

            for lp in range(lp_max + 1):
                for twojp in range(np.abs(2 * lp - 1), 2 * lp + 2, 2):
                    for twomp in range(-twojp, twojp + 1, 2):
                        if idxp == idx:
                            return n_p, lp, twojp, twomp
                        idxp += 1

        raise ValueError("The index provided is too large")
    

    # This is probably mega-slow for larger systems and will need to be numbad
    def matrix_ndflatten(self, matrix, dim=2):
        shape_tuple = (self.num_states,) * dim
        flat_matrix = np.zeros(shape_tuple)

        it = np.nditer(matrix, flags=['multi_index'])

        n, l, twoj_idx, twoj = np.zeros(dim, dtype=int), np.zeros(dim, dtype=int), np.zeros(dim, dtype=int), np.zeros(dim, dtype=int)
        idx = np.zeros(dim, dtype=int)

        for el in it:
            for d in range(dim):
                n[d], l[d], twoj_idx[d] = it.multi_index[d * 3 : (d + 1) * 3]
                twoj[d] = 2 * l[d] - 1 + twoj_idx[d] * 2

            # Special unphysical case: FIXME
            if np.any(twoj < 0):
                continue

            for twom in product(*[range(-twoj[d], twoj[d] + 1, 2) for d in range(dim)]):
                for d in range(dim):
                    idx[d] = self.index_flattener(n[d], l[d], twoj[d], twom[d])

                #print(idx, el, n, l, twoj, twom)
                flat_matrix[tuple(idx)] = el

        # for el in it:

        #     n1, l1, twoj1_idx = it.multi_index[0:3] 
        #     n2, l2, twoj2_idx = it.multi_index[3:6]
        #     twoj1 = 2 * l1 - 1 + twoj1_idx * 2
        #     twoj2 = 2 * l2 - 1 + twoj2_idx * 2

        #     # Special unphysical case:
        #     if (twoj1 == -1 and l1 == 0) or (twoj2 == -1 and l2 == 0):
        #         continue
                
        #     print(n1, n2, l1, l2, twoj1, twoj2)
        #     for twom1 in range(-twoj1, twoj1 + 1, 2):
        #         for twom2 in range(-twoj2, twoj2 + 1, 2):
        #             idx1 = self.index_flattener(n1, l1, twoj1, twom1)
        #             idx2 = self.index_flattener(n2, l2, twoj2, twom2)

        #             flat_matrix[idx1, idx2] = el
        
        return flat_matrix
        




if __name__ == '__main__':
    system = System(Ne_max=4, l_max=0, omega=3, mass=1)

    # idx = system.index_flattener(0, 0, 1, 1)
    # print(idx)
    # print(system.index_unflattener(idx))

    # print(system.index_unflattener(0))
    # print(system.index_unflattener(1))
    # print(system.index_unflattener(2))
    # print(system.index_unflattener(3))
    # print(system.index_unflattener(4))
    # print(system.index_unflattener(5))
    # print(system.index_flattener(2, 0, 1, -1))

    #print(system.num_states)

    obme = system.get_one_body_matrix_elements()
    print(obme.shape)
    print(obme[:,0,1,:,0,1])


    fl_obme = system.matrix_ndflatten(obme)
    print(fl_obme)