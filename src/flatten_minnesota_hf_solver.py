import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from copy import deepcopy
import time
from itertools import product
np.set_printoptions(precision=4, suppress=True, linewidth=200)

import full_minnesota_hf_system as hfs
import full_minnesota_numba as hfs_numba
import hf_numba_helpers
import sys
sys.path.append('../tests')
import on_the_fly # type: ignore


class Solver:

    def __init__(self, system, num_particles=2):
        self.num_particles = num_particles

        # Class containing the model information
        self.system = system

        self.include_m = system.include_m
        self.num_states, self.true_num_states, self.n_level_max, self.l_level_max =\
              system.num_states, system.true_num_states, system.n_level_max, system.l_level_max
        
        self.occupation_matrix = self.get_occupation_matrix()

        # Pick the hole states as those with the lowest energies
        self.original_energies, self.original_energies_flat, self.original_quantum_numbers =\
                                    system.eigenenergies, system.eigenenergies_flat, system.quantum_numbers
        # self.hole_states = np.sort(system.energy_indices[:self.num_particles])
        # self.particle_states = np.sort(system.energy_indices[self.num_particles:])

        # print("Hole states: {}".format(self.hole_states))
        # print("Particle states: {}".format(self.particle_states))


    # We are going to flatten every matrix like a bulldozer
    def run(self, max_iterations=1000, tolerance=1e-8, mixture=0):
        # Hartee-Fock single-particle hamiltonian
        h = np.zeros((self.num_states, self.num_states))

        # Single-particle energies
        sp_energies_flat = self.original_energies_flat
        sp_energies_prev = np.zeros((self.num_states, 1))
        quantum_numbers = self.original_quantum_numbers

        # D (overlap) matrix (transformation between original basis and HF basis, components D_alpha,q) (initial guess)
        # Same structure as h
        D = np.eye(self.num_states)
        prev_D = deepcopy(D)
        rho_prev = np.diag(self.occupation_matrix)

        # First, get 1-body and 2-body matrix elements in the original basis (only need to do this once!)
        t_matrix = self.system.matrix_ndflatten(self.system.get_one_body_matrix_elements(), dim=2, include_m=self.include_m, m_diagonal=True)
        V_matrix_qn, V_non_asym_qn, self.cp_J_mats = self.system.get_two_body_matrix_elements_2()

        # start = time.time()
        # V_non_asym = self.system.matrix_ndflatten(V_non_asym_qn, dim=4, include_m=self.include_m, m_diagonal=True, asym=True)
        # V_matrix = self.system.matrix_ndflatten(V_matrix_qn, dim=4, include_m=self.include_m, m_diagonal=True, asym=True)
        # print("Time to flatten V matrix:", time.time() - start)

        start = time.time()
        V_non_asym = hfs_numba.n_matrix_4dflatten(V_non_asym_qn, False, False, False, self.num_states, self.system.Ne_max, self.l_level_max)
        V_matrix = hfs_numba.n_matrix_4dflatten(V_matrix_qn, False, False, False, self.num_states, self.system.Ne_max, self.l_level_max)
        print("Time to numba flatten V matrix:", time.time() - start)
        # print("ASDSDAD", np.allclose(V_matrix, V_matrix_2))
        # exit()

        #V_matrix = np.zeros((self.num_states, self.num_states, self.num_states, self.num_states))
        #V_matrix = V_non_asym

        # V_matrix_2 = V_non_asym - V_non_asym.transpose(0, 1, 3, 2)
        # print("ASDSDAD", np.allclose(V_matrix, V_matrix_2))

        # TESTS:
        # for b in range(3):
        #     print(V_matrix[0, b, :, b])
        #     # print(V_matrix[:, b, 0, b])
        #     # print(V_non_asym[0, b, :, b])
        #     # print(V_non_asym[:, b, 0, b])

        #     self.system.index_unflattener(b)
        #     n1, l1, twoj1 = self.system.index_unflattener(0)
        #     n2, l2, twoj2 = self.system.index_unflattener(b)
        #     n4, l4, twoj4 = self.system.index_unflattener(b)
        #     twoj1_idx = 0 if twoj1 < 2*l1 else 1
        #     twoj2_idx = 0 if twoj2 < 2*l2 else 1
        #     twoj4_idx = 0 if twoj4 < 2*l4 else 1

        #     for k in range(self.num_states):
        #         if k==b:
        #             continue

        #         n3, l3, twoj3 = self.system.index_unflattener(k)
        #         twoj3_idx = 0 if twoj3 < 2*l3 else 1

        #         print("n1, l1, twoj1:", n1, l1, twoj1)
        #         print("n2, l2, twoj2:", n2, l2, twoj2) 
        #         print("n3, l3, twoj3:", n3, l3, twoj3) 
        #         print(V_matrix[0, b, k, b])
        #         print(V_matrix_qn[n1, l1, twoj1_idx, n2, l2, twoj2_idx, n3, l3, twoj3_idx, n4, l4, twoj4_idx])
        #         # print(V_non_asym[0, b, k, b])
        #         # print(V_non_asym_qn[n1, l1, twoj1_idx, n2, l2, twoj2_idx, n3, l3, twoj3_idx, n4, l4, twoj4_idx])
        #         # print(V_non_asym_qn[n1, l1, twoj1_idx, n2, l2, twoj2_idx, n4, l4, twoj4_idx, n3, l3, twoj3_idx])
        #         for twoJ in range(0, twoj2 + 1):
        #             print("twoJ:", twoJ)
        #             print(self.cp_J_mats[0][n1, l1, twoj1_idx, n2, l2, twoj2_idx, n3, l3, twoj3_idx, n4, l4, twoj4_idx, twoJ])
        #             print(self.cp_J_mats[0][n1, l1, twoj1_idx, n2, l2, twoj2_idx, n4, l4, twoj4_idx, n3, l3, twoj3_idx, twoJ])
        #             print(self.cp_J_mats[0][n4, l4, twoj4_idx, n3, l3, twoj3_idx, n1, l1, twoj1_idx, n2, l2, twoj2_idx, twoJ])
        
        # exit()


        #print(V_matrix[0,2,:,:])
        #print(t_matrix)

        # TESTS:
        # Is t_matrix hermitian?
        print("T hermitian:", np.allclose(t_matrix, t_matrix.conj().T))

        # Is V_matrix symmetric under swaping (1,2)<->(3,4)?
        # all_V_sym = True
        # for k1 in range(self.num_states):
        #     for k2 in range(self.num_states):
        #         herm = np.allclose(V_matrix[k1,k2,:,:], V_matrix[:,:,k1,k2], atol=1e-8)

        #         if not herm:
        #             print("NOT SYMMETRIC UNDER (1,2)<->(3,4)! k1, k2:", k1, k2)
        #             print(V_matrix[k1,k2,:,:] - V_matrix[:,:,k1,k2])
        #             #print(V_matrix[:,:,n1,n2])
        #             all_V_sym = False

        # print("V symmetric under (1,2)<->(3,4):", all_V_sym)

        # Is V_matrix antisymmetric in the last two indices?
        all_V_antisym = True
        for k1 in range(self.num_states):
            for k2 in range(self.num_states):
                antisym = np.allclose(V_matrix[k1,k2,:,:], -V_matrix[k1,k2,:,:].T, atol=1e-8)

                if not antisym:
                    print("NOT ASYM! n1, n2:", k1, k2)
                    print(system.index_unflattener(k1), system.index_unflattener(k2))
                    print(system.index_unflattener(1), system.index_unflattener(0))
                    #print(V_matrix[n1,n2,:,:] + V_matrix[n1,n2,:,:].T)
                    print(V_matrix[k1,k2,:,:])
                    #print(-V_matrix[n1,n2,:,:].T)
                    all_V_antisym = False
        
        print("V antisymmetric in the exchange of last two indices:", all_V_antisym)

        # --------------------------------------------------------------------------------------------
        
        # Iteration of the Hartree-Fock method
        print("ORIGINAL HO ENERGIES:", sp_energies_flat)
        print("HO QUANTUM NUMBERS:", quantum_numbers)
        for i in range(max_iterations):
            print('-----------------------------------')
            print("Iteration", i + 1)

            # Construct the density matrix in the original basis 
            # (in the first iteration, it will be non-zero only for the hole states)
            # We can mix the density matrix from the previous iteration with the current one
            rho = self.get_density_matrix(D) * (1 - mixture) + rho_prev * mixture
            rho_prev = deepcopy(rho)

            # Construct the single-particle hamiltonian in the original basis
            hamiltonian = self.get_hamiltonian(t_matrix, V_matrix, rho)
            gamma = hamiltonian - t_matrix

            # Tests:
            on_the_fly.test_run_matrices(self.system, rho=rho, D=D, hamiltonian=hamiltonian, gamma=gamma, include_m=self.include_m)


            #print(self.occupation_matrix)
            # print("D\n", D[:6, :6])
            print("RHO\n", rho[:6, :6])
            # print("Hamiltonian\n", hamiltonian[:6, :6])

            #print(D[0:8, 0:8])
            # print(rho[-8:, -8:])


            # Numerical stabilization: 
            # 1. the hamiltonian will slowly lose hermiticity due to numerical error. Here we restore it
            hamiltonian_new = (hamiltonian + hamiltonian.conj().T) / 2
            # Make sure the correction is small
            herm_correction = np.linalg.norm(hamiltonian_new - hamiltonian)
            print("H hermiticity correction:", herm_correction)
            if herm_correction > 1e-8 * hamiltonian.size:
                print("H hermiticity correction too large!")
                #exit()
            hamiltonian = hamiltonian_new
            
            
            # Diagonalize the single-particle hamiltonian (automatically ordered by energy)
            sp_energies, eigenvectors = eigh(hamiltonian)
            #print("spe1", np.dot(hamiltonian, eigenvectors[:, 0]), eigenvectors[:, 0])
        

            # Stop iterating if difference between previous energies is smaller than tolerance
            norm = np.linalg.norm(sp_energies - sp_energies_prev) / self.num_states
            if norm < tolerance or i == max_iterations - 1:
                print("")
                print("~~~~~~~~~~~~~~~~~~~~~~ FULL SOLVER ~~~~~~~~~~~~~~~~~~~~~~~~")
                print("----------------------------------------------")
                if norm < tolerance:
                    print("Convergence reached after {} iterations".format(i + 1))
                    print("Single particle energies: {}".format(sp_energies))
                else:
                    print("NO CONVERGENCE AFTER: {} iterations".format(max_iterations))
                    print("Single particle energies: {}".format(sp_energies))

                print("NORM", norm)
                full_sp_energies = self.get_full_sp_energies(sp_energies)
                hf_energy = self.compute_hf_energy(sp_energies, full_sp_energies, t_matrix, V_matrix, V_non_asym, rho,
                                                     D, self.num_particles, self.occupation_matrix)
                return hf_energy, sp_energies, eigenvectors
            
            # Update the previous energies and the D matrix
            sp_energies_prev = deepcopy(sp_energies)
            
            #print(eigenvectors[:6, :6])
            D = eigenvectors
            prev_D = deepcopy(D)

            #print(np.dot(D, D.T.conj())[:6, :6])

            print("Sp_energies:", sp_energies)
            #print("Full sp energies:", self.get_full_sp_energies(sp_energies))

        # If we get here, we have reached the maximum number of iterations
        print("No convergence reached after {} iterations".format(max_iterations))
        return None, None, None


    def get_density_matrix(self, D):
        # Each element of the density matrix can be calculated as:
        # rho_alpha,beta = sum_i D_alpha,i * D*_beta,i = sum_i D_alpha,i * D^dagger_i,beta 
        # (where i runs over the hole states, i.e., up to the number of particles)
        # So we can write:

        if self.include_m:
            trunc_D = D[:, :self.num_particles]
            trunc_D_dagger = trunc_D.conj().T
            rho = np.dot(trunc_D, trunc_D_dagger)

        D_dagger = D.conj().T
        rho = np.einsum('ab,bc,b->ac', D, D_dagger, self.occupation_matrix)

        return rho

    # Return the single particle hamiltonian
    
    def get_hamiltonian(self, t_matrix, V_matrix, rho):
        # There is a conflict between the notes and the PDF -> not sure if contraction is over indices 2, 4 or 4, 2...
        # Gamma is a contraction over mu, sigma of: V_alpha,sigma,beta,mu * rho_mu,sigma
        gamma = np.einsum('abcd,db->ac', V_matrix, rho)

        # Tester:
        # raw_V, rawV_na, _ = self.system.get_two_body_matrix_elements()

        # gamma_t = np.zeros((self.num_states, self.num_states))
        # for k1 in range(self.num_states):
        #     for k2 in range(self.num_states):
        #         for mu in range(self.num_states):
        #             for sigma in range(self.num_states):
        #                 n1, l1, twoj1 = self.system.index_unflattener(k1)
        #                 n2, l2, twoj2 = self.system.index_unflattener(sigma)
        #                 n3, l3, twoj3 = self.system.index_unflattener(k2)
        #                 n4, l4, twoj4 = self.system.index_unflattener(mu)
        #                 twoj1_idx = 0 if twoj1 < 2*l1 else 1
        #                 twoj2_idx = 0 if twoj2 < 2*l2 else 1
        #                 twoj3_idx = 0 if twoj3 < 2*l3 else 1
        #                 twoj4_idx = 0 if twoj4 < 2*l4 else 1

        #                 if l1 != l2 or l3 != l4:
        #                     continue
        #                 elif twoj1 != twoj2 or twoj3 != twoj4:
        #                     continue

        #                 vmat = V_matrix[k1, sigma, k2, mu]
        #                 gamma_t[k1, k2] += vmat * rho[mu, sigma]

        #                 v_val = raw_V[n1, l1, twoj1_idx, n2, l2, twoj2_idx, n3, l3, twoj3_idx, n4, l4, twoj4_idx]

        #                 if not np.isclose(v_val, vmat):
        #                     print("SDADSD. Raw, Flattened", v_val, vmat)
                        # l, l_prime = l1, l3
                        # twoj, twoj_prime = twoj1, twoj3
                        # twoj_idx, twoj_prime_idx = twoj1_idx, twoj3_idx
                        
                        # matel1 = 0
                        # matel2 = 0
                        # for twoJ in range(np.abs(twoj - twoj_prime), twoj + twoj_prime + 1, 2):
                        #     matel1 += self.cp_J_mats[0][n1, l, twoj_idx, n2, l_prime, twoj_prime_idx, n3, l, twoj_idx, n4, l_prime, twoj_prime_idx, twoJ] *\
                        #             (twoJ + 1) / ((twoj + 1) * (twoj_prime + 1))
                        #     matel2 += self.cp_J_mats[0][n1, l, twoj_idx, n2, l_prime, twoj_prime_idx, n4, l_prime, twoj_prime_idx, n3, l, twoj_idx, twoJ] *\
                        #             (twoJ + 1) / ((twoj + 1) * (twoj_prime + 1))
                        # matel_asym = matel1 - matel2

                        # if not np.isclose(matel_asym, vmat):
                        #     print("LOLO", vmat, matel_asym)
                        # elif matel_asym != 0:
                        #     print("LILI", vmat, matel_asym)
                        #     print(n1, l1, twoj1, n2, l2, twoj2, n3, l3, twoj3, n4, l4, twoj4)


        #print("CONSGAM", np.allclose(gamma, gamma_t))
        # exit()
        
        # print("GAMMA")
        # print(gamma)


        return t_matrix + gamma
    

    def get_occupation_matrix(self):
        occ_matrix = np.zeros((self.num_states))
        if not self.include_m:
            count = 0
            for k in range(self.num_states):
                n, l, twoj = self.system.index_unflattener(k)
                occ_j = twoj + 1
                occ_matrix[k] = occ_j
                count += occ_j

                if count > self.num_particles:
                    occ_matrix[k] -= count - self.num_particles
                    count = self.num_particles
                    return occ_matrix
        
        else:
            for k in range(self.num_states):
                occ_matrix[k] = 1
        
        return occ_matrix
    

    def get_full_sp_energies(self, sp_energies):
        if self.include_m:
            return sp_energies

        full_sp_energies = []
        for k in range(self.num_states):
            n, l, twoj = self.system.index_unflattener(k)
            multiplicity = twoj + 1
            full_sp_energies.extend([sp_energies[k]] * multiplicity)
        
        return np.array(full_sp_energies)
    


    @staticmethod
    # TODO: Do this properly using the occupation matrix... this is a disaster right now
    def compute_hf_energy(sp_energies, full_sp_energies, t_matrix, V_matrix, V_non_asym, rho, D, n_particles, occ_matrix):
        
        #D = D.T
        # Change the basis of the matrices to the HF basis
        D_dagger = D.conj().T

        rho_hf = np.dot(D_dagger, np.dot(rho, D))
        print("rho_hf", rho_hf)
        t_matrix_hf = np.dot(D_dagger, np.dot(t_matrix, D))

        start = time.time()
        V_matrix_prov = np.einsum('abcd,ea,df->ebcf', V_matrix, D_dagger, D, optimize='optimal')
        V_matrix_hf = np.einsum('ebcf,gb,ch->eghf', V_matrix_prov, D_dagger, D, optimize='optimal')

        V_non_asym_prov = np.einsum('abcd,ea,df->ebcf', V_non_asym, D_dagger, D, optimize='optimal')
        V_non_asym_hf = np.einsum('ebcf,gb,ch->eghf', V_non_asym_prov, D_dagger, D, optimize='optimal')
        print("Time to change basis (V):", time.time() - start)

        #testV = hf_numba_helpers.n_change_basis_4d(V_matrix, D)
 
        # Compute the energy in multiple ways:
        #hf_energy = 0.5 * (np.trace(t_matrix) + np.sum(full_sp_energies[:n_particles]))
        hf_energy = 0.5 * (np.einsum('ii,i', t_matrix_hf, occ_matrix) + np.sum(full_sp_energies[:n_particles]))
        
        # print("HF ENERGY", hf_energy)
        # exit()
        # return hf_energy

        # These two are wrong (has to be with the non-asym potential)
        e_hartree = 0.5 * np.einsum('ijkl,ki,lj', V_non_asym_hf, rho_hf, rho_hf, optimize='optimal')
        e_fock = -0.5 * np.einsum('ijlk,ki,lj', V_non_asym_hf, rho_hf, rho_hf, optimize='optimal')
        hf_energy_2 = np.sum(full_sp_energies[:n_particles]) - e_hartree - e_fock

        e_ho = np.einsum('ij,ji', t_matrix_hf, rho_hf)
        e_int = 0.5 * np.einsum('ijkl,ki,lj', V_matrix_hf, rho_hf, rho_hf, optimize='optimal')
        e_int_2 = hf_energy - e_ho

        # Another way of getting the energy
        hf_energy_3 = np.einsum('ii,i', t_matrix_hf, occ_matrix) + 0.5 * np.einsum('ijij,i,j', V_matrix_hf, occ_matrix, occ_matrix)
        hf_energy_4 = np.sum(full_sp_energies[:n_particles]) - 0.5 * np.einsum('ijij,i,j', V_matrix_hf, occ_matrix, occ_matrix)
        #hf_energy_5 = np.einsum('ab,ba', t_trunc, rho_trunc) + 0.5 * np.einsum('abcd,db,ca', V_trunc, rho_trunc, rho_trunc)
        hf_energy_5 = np.einsum('ab,ba', t_matrix, rho) + 0.5 * np.einsum('abcd,db,ca', V_matrix, rho, rho, optimize='optimal')
        #print(np.trace(t_trunc), 0.5 * np.trace(np.einsum('ijij', V_trunc)))

        # print(t_trunc)
        # print(t_matrix)
    
        print("SDAS", hf_energy, hf_energy_2, hf_energy_3, hf_energy_4, hf_energy_5)
        print("E_hartree", e_hartree)
        print("E_fock", e_fock)
        print("E_onebody", e_ho)
        print("E_int", e_int, e_int_2)

        return hf_energy
    


if __name__ == "__main__":

    system = hfs.System(Ne_max=2, l_max=1, hbar_omega=10, mass=939, include_m=False)
    solver = Solver(system, num_particles=8)

    print("Number of states:", system.num_states)

    start_time = time.time()
    hf_energy, sp_energies, eigenvectors = solver.run(max_iterations=500, mixture=0.3)

    end_time = time.time()
    print("Time elapsed: {} seconds".format(end_time - start_time))

    print("Hartree-Fock energy: {} MeV".format(hf_energy))

    print(system.num_states)

    # wf = lambda r: h3d.wavefunction(r, k=1, l=0, hbar_omega=1, mass=1)

    # r = np.linspace(0, 10, 1000)

    # plt.plot(r, wf(r))
    # plt.show()