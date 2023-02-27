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

HBAR = 197.3269788 # MeV * fm
#HBAR = 1

# Simplified neutron drop system
class System:
    # Minnesota potential parameters
    V0R = 200 # MeV
    V0t = 178 # MeV
    V0s = 91.85 # MeV
    kappaR = 1.487 # fm^-2
    kappat = 0.639 # fm^-2
    kappas = 0.465 # fm^-2 

    def __init__(self, k_num, l_num, hbar_omega=1, mass=1) -> None:
        self.k_num = k_num
        self.l_num = l_num
        self.num_states = k_num * l_num * 2 # times 2 for the spin

        self.hbar_omega = hbar_omega
        self.mass = mass

        # List of all energies and their indices
        self.eigenenergies, self.energy_indices = self.get_lowest_energies(self.num_states)

        self.wavefunctions, self.sqrt_norms = self.generate_wavefunctions()

        # The way the indices will work is idx = k + l * k_num + spin * k_num * l_num
        # (where here spin is 0 or 1)
    

    def generate_wavefunctions(self):
        Ne_max = 2 * self.k_num
        wavefunctions = np.zeros((Ne_max * 2 + 1, Ne_max * 2 + 1), dtype=object)
        sqrt_norms = np.zeros((Ne_max * 2 + 1, Ne_max * 2 + 1), dtype=np.float64)
        for k in range(Ne_max * 2 + 1):
            for l in range(Ne_max * 2 + 1):

                r = np.linspace(0, 15, 2500)
                _, sqrt_norm = h3d.series_wavefunction(r, k=k, l=l, hbar_omega=self.hbar_omega, mass=self.mass)

                wavefunctions[k, l] = lambda r, k=k, l=l: h3d.wavefunction(r, k=k, l=l, hbar_omega=self.hbar_omega, mass=self.mass)
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
    
    

    def grid_twod_radial_integral(self, V0, mu, k1, k2, k3, k4, integration_limit=10):
        r1 = np.linspace(1e-8, integration_limit, 700)
        r2 = np.linspace(1e-8, integration_limit, 700)

        rect = np.diff(r1)[0] * np.diff(r2)[0]

        r1, r2 = np.meshgrid(r1, r2, indexing='ij')

        # TODO: Shouldn't 1 and 4 carry r2?
        rfunc_1 = self.wavefunctions[k1, 0](r2)
        rfunc_2 = self.wavefunctions[k2, 0](r1)
        rfunc_3 = self.wavefunctions[k3, 0](r2)
        rfunc_4 = self.wavefunctions[k4, 0](r1)

        # TODO: is this 0.5 or 1? and + or -?
        exp1 = np.exp(-mu * (r1 + r2)**2)
        exp3 = np.exp(-mu * (r1 - r2)**2)
        potential_l0 = -0.5 * V0 * 1/(2 * mu) * np.reciprocal(r1 * r2) * (-exp1 + exp3)
        # Remove all infs and nans
        # potential_l0[np.isnan(potential_l0)] = 0
        # potential_l0[np.isinf(potential_l0)] = 0

        # plt.matshow(np.log(-potential_l0))
        # plt.colorbar()
        # plt.show()

        #t0 = time.time()
        radial_integral = np.sum(r1**2 * r2**2 * rfunc_1 * rfunc_2 * potential_l0 * rfunc_3 * rfunc_4) * rect
        radial_integral *= np.pi**2/4 #1/ (4 * np.pi)
        #t1 = time.time()
        #print(f"GRID Time to compute integral: {t1 - t0}, Value: {radial_integral}")

        return radial_integral


    #@jit TODO: figure out how to get this to work
    def numba_twod_radial_integral(self, V0, mu, k1, sqn1, k2, sqn2, k3, sqn3, k4, sqn4, integration_limit=10):
        numba_LLC = LowLevelCallable(self.tbme_integrand.ctypes)

        t0 = time.time()
        radial_integral, error = dblquad(numba_LLC, 0, integration_limit, 0, integration_limit,
                                        args=(V0, mu, k1, sqn1, k2, sqn2, k3, sqn3, k4, sqn4, self.hbar_omega, self.mass))
        radial_integral *= 1 / (4 * np.pi)
        t1 = time.time()
        #print(f"Time to compute integral: {t1 - t0}, Value: {radial_integral}")

        return radial_integral

    # # Build the function to integrate as a cfunc
    # @cfunc(float64(intc, CPointer(float64)))
    # def tbme_integrand(n, args):
    #     r1, r2, V0, mu, k1, sqn1, k2, sqn2, k3, sqn3, k4, sqn4, hbar_omega, mass =\
    #          (args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13])

    #     wf1_val = h3d.series_wavefunction_val(r1, k1, 0, hbar_omega, mass, sqn1)
    #     wf2_val = h3d.series_wavefunction_val(r2, k2, 0, hbar_omega, mass, sqn2)
    #     wf3_val = h3d.series_wavefunction_val(r1, k3, 0, hbar_omega, mass, sqn3)
    #     wf4_val = h3d.series_wavefunction_val(r2, k4, 0, hbar_omega, mass, sqn4)

    #     func_val = r1**2 * r2**2 * wf1_val * wf2_val * c_potential_l0(V0, mu, r1, r2) * wf3_val * wf4_val

    #     return func_val

    
    # Get the one-body matrix elements in the HO basis
    def get_one_body_matrix_elements(self):
        # As given in https://wikihost.nscl.msu.edu/TalentDFT/lib/exe/fetch.php?media=ho_spherical.pdf,
        # the one-body matrix elements for this model in the HO basis are...
        # I'm almost sure the above is wrong. As the particles are trapped in a HO potential, the "kinetic energy"
        # is just the harmonic oscillator energy, and the OBME should be diagonal in the HO basis.

        t = np.zeros((self.num_states, self.num_states)) 

        for k1 in range(self.k_num):
            for l in range(self.l_num):
                for s in range(0, 2):
                    for k2 in range(self.k_num):
                        idx1 = self.get_idx(k1, l, s)
                        idx2 = self.get_idx(k2, l, s)

                        if k1 == k2:
                            t[idx1, idx2] = 2 * k1 + l + 3/2
                        else:
                            t[idx1, idx2] = 0

                        # elif k1 == k2 - 1:
                        #     t[idx1, idx2] = np.sqrt(k2 * (k2 + l + 1/2))
                        # elif k1 == k2 + 1:
                        #     t[idx1, idx2] = np.sqrt(k1 * (k1 + l + 1/2))
                        # else:
                        #     t[idx1, idx2] = 0

        t *= 0.5 * self.hbar_omega

        return t
    

    # Get the antisymmetrized two-body matrix elements in the HO basis
    def get_two_body_matrix_elements(self):
        num_states = self.num_states
        V_mat = np.zeros((num_states, num_states, num_states, num_states))
        
        # From https://wikihost.nscl.msu.edu/TalentDFT/lib/exe/fetch.php?media=hf_truncated_v2.pdf, we can evaluate the matrix elements
        # for the Minnesota potential in the HO basis as...

        max_index = self.get_idx(self.k_num - 1, 0, 1)

        count = 0
        total = self.k_num**4 #* 2**4
        for k1, k2, k3, k4 in product(range(self.k_num), repeat=4):
        
            count += 1
            if count % 100 == 0:
                print("{}/{}".format(count, total))

            # We do this to antisymmetrize the matrix correctly
            if k3 > k4: 
                continue

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

            # # Get the integrals! (only depend on k!)
            vr_exp = self.grid_twod_radial_integral(self.V0R, self.kappaR, k1, k2, k3, k4)
            vs_exp = self.grid_twod_radial_integral(-self.V0s, self.kappas, k1, k2, k3, k4)

            vr_exp_a = self.grid_twod_radial_integral(self.V0R, self.kappaR, k1, k2, k4, k3)
            vs_exp_a = self.grid_twod_radial_integral(-self.V0s, self.kappas, k1, k2, k4, k3)

            VD_exp = 0.5 * (vr_exp + vs_exp)
            VEPr_exp = 0.5 * (vr_exp_a + vs_exp_a)

            V_exp = VD_exp + VEPr_exp

            # Element is spin-independent:
            idx1_p, idx2_p, idx3_p, idx4_p = self.get_idx(np.array([k1, k2, k3, k4]), l=0, s=1)

            # Should some of these be 0? Only if the potential is spin-diagonal, which it probably isn't...
            for idx1n, idx2n, idx3n, idx4n in product([idx1, idx1_p], [idx2, idx2_p], [idx3, idx3_p], [idx4, idx4_p]):
                _, s1 = self.get_kls(idx1n)
                _, s2 = self.get_kls(idx2n)
                _, s3 = self.get_kls(idx3n)
                _, s4 = self.get_kls(idx4n)

                # Spin deltas:
                d1 = 1 if (s1 == s3 and s2 == s4) else 0
                d2 = 1 if (s1 == s4 and s2 == s3) else 0
                if d1 - d2 == 0:
                    continue
                else:
                    sign = d1 - d2

                # Remember to add the conjugate...
                V_mat[idx1n, idx2n, idx3n, idx4n] = sign * V_exp
                V_mat[idx3n, idx4n, idx1n, idx2n] = sign * V_exp.conjugate()

                # And to antisymmetrize:
                V_mat[idx1n, idx2n, idx4n, idx3n] = -V_mat[idx1n, idx2n, idx3n, idx4n]
                V_mat[idx4n, idx3n, idx1n, idx2n] = -V_mat[idx3n, idx4n, idx1n, idx2n]


        return V_mat


    def get_idx(self, k, l, s):
        #return k + l * self.k_num + s * self.k_num * self.l_num
        return 2 * k + s
    
    def get_kls(self, idx):
        k = idx // 2
        s = idx % 2
        return k, s

    # Energies of the HO basis eigenstates
    def get_ho_energies(self, k, l):
        return 0.5 * self.hbar_omega * (2 * k + l + 3/2)

    # Get the indices of the lowest energy states
    def get_lowest_energies(self, num):
        energies = np.zeros(self.num_states)
        for k in range(self.k_num):
            for l in range(self.l_num):
                for s in range(2):
                    idx = self.get_idx(k, l, s)
                    energies[idx] = self.get_ho_energies(k, l)

        idx = np.argsort(energies)[:num]
        return energies, idx



if __name__ == '__main__':
    pass