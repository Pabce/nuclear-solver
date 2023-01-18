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

HBAR = 1

# Simplified neutron drop system
class System:
    # Minnesota potential parameters
    V0R = 200 # MeV
    V0t = 178 # MeV
    V0s = 91.85 # MeV
    kappaR = 1.487 # fm^-2
    kappat = 0.639 # fm^-2
    kappas = 0.465 # fm^-2 

    def __init__(self, k_num, l_num, omega=1, mass=1) -> None:
        self.k_num = k_num
        self.l_num = l_num
        self.n_states = k_num * l_num * 2 # times 2 for the spin

        self.omega = omega
        self.mass = 1

        # List of all energies and their indices
        self.eigenenergies, self.energy_indices = self.get_lowest_energies(self.n_states)

        self.wavefunctions, self.sqrt_norms = self.generate_wavefunctions()

        # The way the indices will work is idx = k + l * k_num + spin * k_num * l_num
        # (where here spin is 0 or 1)
    

    def generate_wavefunctions(self):
        wavefunctions = np.zeros((self.k_num, self.l_num), dtype=object)
        sqrt_norms = np.zeros((self.k_num, self.l_num), dtype=np.float64)
        for k in range(self.k_num):
            for l in range(self.l_num):

                r = np.linspace(0, 15, 2500)
                _, sqrt_norm = h3d.series_wavefunction(r, k=k, l=l, omega=self.omega, mass=self.mass)

                wavefunctions[k, l] = lambda r, k=k, l=l: h3d.wavefunction(r, k=k, l=l, omega=self.omega, mass=self.mass)
                sqrt_norms[k, l] = sqrt_norm

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
    
    @staticmethod
    # Integral for a potential of the form V(r) = V0 * exp(-mu * (r1-r2)**2))
    def two_dimensional_radial_integral(V0, mu, rfunc_1, rfunc_2, rfunc_3, rfunc_4, integration_limit=10):

        # l=0 component of the multipole expansion of the potential
        potential_l0 = lambda x1, x2: - 0.5 * V0 / (2 * mu * x1 * x2) * np.exp(-mu * (x1 + x2)**2) * (-1 + np.exp(4 * mu * x1 * x2))

        func = lambda x1, x2: x1**2 * x2**2 * rfunc_1(x1) * rfunc_2(x2) * potential_l0(x1, x2) * rfunc_3(x1) * rfunc_4(x2)

        # There is some numerical prefactor I have to figure out here (is it 4pi?)
        
        t0 = time.time()
        radial_integral, error = dblquad(func, 0, integration_limit, 0, integration_limit)
        radial_integral *= 1/ (4 * np.pi)
        t1 = time.time()
        print(f"Time to compute integral: {t1 - t0}, Value: {radial_integral}")

        return radial_integral
    

    def grid_twod_radial_integral(self, V0, mu, k1, k2, k3, k4, integration_limit=10):
        r1 = np.linspace(0.001, integration_limit, 500)
        r2 = np.linspace(0.001, integration_limit, 500)

        rect = np.diff(r1)[0] * np.diff(r2)[0]

        r1, r2 = np.meshgrid(r1, r2, indexing='ij')

        rfunc_1 = self.wavefunctions[k1, 0](r1)
        rfunc_2 = self.wavefunctions[k2, 0](r2)
        rfunc_3 = self.wavefunctions[k3, 0](r1)
        rfunc_4 = self.wavefunctions[k4, 0](r2)

        potential_l0 = - 0.5 * V0 * 1/(2 * mu) * np.reciprocal(r1 * r2) * np.exp(-mu * (r1 + r2)**2) * (-1 + np.exp(4 * mu * r1 * r2))

        #t0 = time.time()
        radial_integral = np.sum(r1**2 * r2**2 * rfunc_1 * rfunc_2 * potential_l0 * rfunc_3 * rfunc_4) * rect
        radial_integral *= 1/ (4 * np.pi)
        #t1 = time.time()
        #print(f"GRID Time to compute integral: {t1 - t0}, Value: {radial_integral}")

        return radial_integral


    #@jit TODO: figure out how to get this to work
    def numba_twod_radial_integral(self, V0, mu, k1, sqn1, k2, sqn2, k3, sqn3, k4, sqn4, integration_limit=10):
        numba_LLC = LowLevelCallable(self.tbme_integrand.ctypes)

        t0 = time.time()
        radial_integral, error = dblquad(numba_LLC, 0, integration_limit, 0, integration_limit,
                                        args=(V0, mu, k1, sqn1, k2, sqn2, k3, sqn3, k4, sqn4, self.omega, self.mass))
        radial_integral *= 1 / (4 * np.pi)
        t1 = time.time()
        #print(f"Time to compute integral: {t1 - t0}, Value: {radial_integral}")

        return radial_integral

    # Build the function to integrate as a cfunc
    @cfunc(float64(intc, CPointer(float64)))
    def tbme_integrand(n, args):
        r1, r2, V0, mu, k1, sqn1, k2, sqn2, k3, sqn3, k4, sqn4, omega, mass =\
             (args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13])

        wf1_val = h3d.series_wavefunction_val(r1, k1, 0, omega, mass, sqn1)
        wf2_val = h3d.series_wavefunction_val(r2, k2, 0, omega, mass, sqn2)
        wf3_val = h3d.series_wavefunction_val(r1, k3, 0, omega, mass, sqn3)
        wf4_val = h3d.series_wavefunction_val(r2, k4, 0, omega, mass, sqn4)

        func_val = r1**2 * r2**2 * wf1_val * wf2_val * c_potential_l0(V0, mu, r1, r2) * wf3_val * wf4_val

        return func_val

    
    # Get the one-body matrix elements in the HO basis
    def get_one_body_matrix_elements(self):
        # As given in https://wikihost.nscl.msu.edu/TalentDFT/lib/exe/fetch.php?media=ho_spherical.pdf,
        # the one-body matrix elements for this model in the HO basis are...

        t = np.zeros((self.n_states, self.n_states)) 

        for k1 in range(self.k_num):
            for l in range(self.l_num):
                for s in range(0, 2):
                    for k2 in range(self.k_num):
                        idx1 = self.get_idx(k1, l, s)
                        idx2 = self.get_idx(k2, l, s)

                        if idx1 == idx2:
                            t[idx1, idx2] = 2 * k1 + l + 3/2
                        elif idx1 == idx2 - 1:
                            t[idx1, idx2] = np.sqrt(k2 * (k2 + l + 1/2))
                        elif idx1 == idx2 + 1:
                            t[idx1, idx2] = np.sqrt(k1 * (k1 + l + 1/2))

        t *= 0.5 * HBAR * self.omega
        
        return t
    

    # Get the antisymmetrized two-body matrix elements in the HO basis
    def get_two_body_matrix_elements(self):
        n_states = self.n_states
        V_mat = np.zeros((n_states, n_states, n_states, n_states))
        
        # From https://wikihost.nscl.msu.edu/TalentDFT/lib/exe/fetch.php?media=hf_truncated_v2.pdf, we can evaluate the matrix elements
        # for the Minnesota potential in the HO basis as...

        max_index = self.get_idx(self.k_num - 1, 0, 1)

        count = 0
        total = self.k_num**4 #* 2**4
        for k1, k2, k3, k4 in product(range(self.k_num), repeat=4):

            count += 1
            if count % 100 == 0:
                print("{}/{}".format(count, total))

            # Get the indices:
            idx1, idx2, idx3, idx4 = self.get_idx(np.array([k1, k2, k3, k4]), l=0, s=0)

            # We can skip if not in the "upper triangle":
            if idx1 * max_index + idx2 < idx3 * max_index + idx4: 
                continue

            # Spin deltas:
            # d1 = 1 if (s1 == s3 and s2 == s4) else 0
            # d2 = 1 if (s1 == s4 and s2 == s3) else 0
            # if d1 - d2 == 0:
            #     continue
                
            # Get the wavefunctions:
            # w1, sqn1 = self.wavefunctions[k1, 0], self.sqrt_norms[k1, 0]
            # w2, sqn2 = self.wavefunctions[k2, 0], self.sqrt_norms[k2, 0]
            # w3, sqn3 = self.wavefunctions[k3, 0], self.sqrt_norms[k3, 0]
            # w4, sqn4 = self.wavefunctions[k4, 0], self.sqrt_norms[k4, 0]

            # # Get the integrals! (only depend on k!)
            # # There's something fishy going on here...
            # vr_exp = self.numba_twod_radial_integral(self.V0R, self.kappaR, k1, sqn1, k2, sqn2, k3, sqn3, k4, sqn4)
            # vs_exp = self.numba_twod_radial_integral(-self.V0s, self.kappas, k1, sqn1, k2, sqn2, k3, sqn3, k4, sqn4)
            vr_exp = self.grid_twod_radial_integral(self.V0R, self.kappaR, k1, k2, k3, k4)
            vs_exp = self.grid_twod_radial_integral(-self.V0s, self.kappas, k1, k2, k3, k4)

            vr_exp_a = vr_exp #self.numba_twod_radial_integral(self.V0R, self.kappaR, k1, sqn1, k2, sqn2, k4, sqn4, k3, sqn3)
            vs_exp_a = vs_exp #self.numba_twod_radial_integral(-self.V0s, self.kappas, k1, sqn1, k2, sqn2, k4, sqn4, k3, sqn3)

            VD_exp = 0.5 * (vr_exp + vs_exp)
            VEPr_exp = 0.5 * (vr_exp_a + vs_exp_a)

            V_exp = VD_exp + VEPr_exp

            V_mat[idx1, idx2, idx3, idx4] = V_exp
            # The hamiltonian is hermitian, so remember to add the conjugate:
            V_mat[idx3, idx4, idx1, idx2] = V_exp.conjugate()

            # Element is spin-independent and matrix is spin-diagonal:
            idx1_p, idx2_p, idx3_p, idx4_p = self.get_idx(np.array([k1, k2, k3, k4]), l=0, s=1)
            V_exp = V_mat[idx1, idx2, idx3, idx4]
            V_mat[idx1_p, idx2_p, idx3_p, idx4_p] = V_exp
            V_mat[idx3_p, idx4_p, idx1_p, idx2_p] = V_exp.conjugate()


        return V_mat


    def get_idx(self, k, l, s):
        return k + l * self.k_num + s * self.k_num * self.l_num

    # Energies of the HO basis eigenstates
    def get_ho_energies(self, k, l):
        return 0.5 * self.omega * (2 * k + l + 3/2)

    # Get the indices of the lowest energy states
    def get_lowest_energies(self, num):
        energies = np.zeros(self.n_states)
        for k in range(self.k_num):
            for l in range(self.l_num):
                for s in range(2):
                    idx = self.get_idx(k, l, s)
                    energies[idx] = self.get_ho_energies(k, l)

        idx = np.argsort(energies)[:num]
        return energies, idx



if __name__ == '__main__':
    pass