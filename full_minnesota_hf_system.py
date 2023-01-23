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
import py3nj
import moshinsky_way as mw

HBAR = 1

# Neutron drop system
# For purposes of computing the matrix elements, and also because
# we are interested in chossing the lowest energy states, we will
# work with the quantum number N = 2n + l, calling it Ne here
# (as it represents the energy of the system)
# I hope I don't regret this
class System:
    # Minnesota potential parameters
    V0R = 200 # MeV
    V0t = 178 # MeV
    V0s = 91.85 # MeV
    kappaR = 1.487 # fm^-2
    kappat = 0.639 # fm^-2
    kappas = 0.465 # fm^-2 

    def __init__(self, Ne_num, l_num, omega=1, mass=1) -> None:
        self.Ne_num = Ne_num
        self.l_num = l_num
        # l num represents the maximum value of l (cannot be larger than Ne_num)
        if l_num > Ne_num:
            raise ValueError("l_num cannot be larger than Ne_num")

        # l can go from Ne to 0 in steps of 2
        n_states = 0
        for Ne in range(Ne_num):
            max_l = min(Ne, l_num)
            if max_l % 2 == 0:
                n_states += max_l / 2 + 1
            else:
                n_states += (max_l + 1) / 2

        self.n_states = n_states * 2 # times 2 for the spin / j

        self.omega = omega
        self.mass = mass

        # List of all energies and their indices
        self.eigenenergies = self.get_lowest_energies()

        self.wavefunctions, self.sqrt_norms = self.generate_wavefunctions()

        # The way the indices will work is idx = k + l * k_num + spin * k_num * l_num
        # (where here spin is 0 or 1)
    

    def generate_wavefunctions(self, r_limit = 15, r_steps = 2500):
        wavefunctions = np.zeros((self.Ne_num, self.l_num), dtype=object)
        sqrt_norms = np.zeros((self.Ne_num, self.l_num), dtype=np.float64)
        for Ne in range(self.Ne_num):
            for l in range(self.l_num):
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
    

    def central_potential_reduced_matrix_element(self, V0, mu, k1, k2, l1, l2, integration_limit=10):
        r = np.linspace(0, integration_limit, 1000)
        # We can compute the reduced matrix element as given by Moshinsky:

        rfunc_1 = self.wavefunctions[k1, l1](r)
        rfunc_2 = self.wavefunctions[k2, l2](r)

        pot = V0 * np.exp(-mu * r**2)

        return np.trapz(rfunc_1 * pot * rfunc_2, r)
    

    def central_potential_ls_coupling_matrix_element(self, V0, mu, k1, k2, k3, k4, l1, l2, l3, l4, lamb):
        pass

    
    # Get the one-body matrix elements in the HO basis
    def get_one_body_matrix_elements(self):
        # As given in https://wikihost.nscl.msu.edu/TalentDFT/lib/exe/fetch.php?media=ho_spherical.pdf,
        # the one-body matrix elements for this model in the HO basis are...
        # These elements only depend on n and l, not on the spin:

        t = np.zeros((self.Ne_num, self.l_num, self.Ne_num, self.l_num), dtype=np.float64)

        # They are also diagonal in l
        for Ne1, Ne2 in product(range(self.Ne_num), repeat=2):
            for l in range(self.l_num):
                n1 = (Ne1 - l) / 2
                n2 = (Ne2 - l) / 2

                if n1 == n2:
                    val = Ne1 + 3/2
                elif n1 == n2 - 1:
                    val = np.sqrt(n2 * (n2 + l + 1/2))
                elif n2 == n2 + 1:
                    val = np.sqrt(n1 * (n1 + l + 1/2))
        
                t[Ne1, l, Ne2, l] = val

        t *= 0.5 * HBAR * self.omega
        
        return t
    

    # Get the antisymmetrized two-body matrix elements in the HO basis
    def get_two_body_matrix_elements(self):
        n_states = self.n_states
        V_mat = np.zeros((n_states, n_states, n_states, n_states))
        
        # 

        return V_mat

    # Energies of the HO basis eigenstates
    def get_ho_energies(self, Ne=-1, n=-1, l=-1):
        if Ne == -1:
            return 0.5 * self.omega * (2 * n + l + 3/2)
        else:
            return 0.5 * self.omega * Ne

    # Get the indices of the lowest energy states
    def get_lowest_energies(self, num):
        energies = np.zeros(self.Ne_num, self.l_num)
        for Ne in range(self.Ne_num):
            for l in range(self.l_num):
                    energies[Ne, l] = self.get_ho_energies(Ne)

        #idx = np.argsort(energies)[:num]
        return energies



if __name__ == '__main__':
    pass