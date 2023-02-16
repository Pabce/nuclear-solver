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


class Solver:

    def __init__(self, system, num_particles=2):
        self.num_particles = num_particles

        # Class containing the model information
        self.system = system

        self.num_states, self.n_level_max, self.l_level_max = system.num_states, system.n_level_max, system.l_level_max

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

        # First, get 1-body and 2-body matrix elements in the original basis (only need to do this once!)
        t_matrix = self.system.matrix_ndflatten(self.system.get_one_body_matrix_elements(), dim=2, m_diagonal=True)
        V_matrix_qn, V_non_asym_qn = self.system.get_two_body_matrix_elements()

        # start = time.time()
        # #V_non_asym_1 = -self.system.matrix_ndflatten(V_non_asym_qn, dim=4, m_diagonal=True, asym=True)
        # V_matrix_1 = self.system.matrix_ndflatten(V_matrix_qn, dim=4, m_diagonal=True, asym=True)
        # print("Time to flatten V matrix:", time.time() - start)

        start = time.time()
        V_non_asym = hfs_numba.n_matrix_4dflatten(V_non_asym_qn, True, False, self.num_states, self.system.Ne_max, self.l_level_max)
        V_matrix = hfs_numba.n_matrix_4dflatten(V_matrix_qn, True, True, self.num_states, self.system.Ne_max, self.l_level_max)
        print("Time to numba flatten V matrix:", time.time() - start)
        #V_matrix = np.zeros((self.num_states, self.num_states, self.num_states, self.num_states))
        #V_matrix = V_non_asym

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

        # TODO: We want to reorder our matrices so that the hole states are first (maybe we don't actually need this)

        print("ORIGINAL HO ENERGIES:", sp_energies_flat)
        print("HO QUANTUM NUMBERS:", quantum_numbers)
        for i in range(max_iterations):
            print('-----------------------------------')
            print("Iteration", i + 1)

            # Construct the density matrix in the original basis (in the first iteration, it will be non-zero only for the hole states)
            rho = self.get_density_matrix(D)

            # Construct the single-particle hamiltonian in the original basis
            hamiltonian = self.get_hamiltonian(t_matrix, V_matrix, rho)
            #print(hamiltonian)

            # TESTS:
            # Is rho equal to its sqaure?
            print("RHO equal to RHO^2:", np.allclose(rho, np.dot(rho, rho)))
            # Is rho hermitian?
            print("RHO hermitian:", np.allclose(rho, rho.conj().T))
            
            # TODO: make sure rho is diagonal in l, j!
            rho_diag_l, rho_diag_lj, rho_diag_ljm = True, True, True
            D_diag_l, D_diag_lj, D_diag_ljm = True, True, True
            hamiltonian_diag_l, hamiltonian_diag_lj, hamiltonian_diag_ljm = True, True, True
            for k1 in range(self.num_states):
                for k2 in range(self.num_states):
                    n1, l1, twoj1, twom1 = system.index_unflattener(k1)
                    n2, l2, twoj2, twom2 = system.index_unflattener(k2)

                    if l1 != l2:
                        if not np.allclose(rho[k1, k2], 0):
                            print(rho[k1, k2])
                            # print("RHO NOT DIAGONAL IN L! k1, k2:", k1, k2)
                            # print(system.index_unflattener(k1), system.index_unflattener(k2))
                            rho_diag_l = False
                        if not np.allclose(D[k1, k2], 0):
                            # print("D NOT DIAGONAL IN L! k1, k2:", k1, k2)
                            # print(D[k1, k2])
                            # print(system.index_unflattener(k1), system.index_unflattener(k2))
                            D_diag_l = False
                        if not np.allclose(hamiltonian[k1, k2], 0):
                            # print("HAMILTONIAN NOT DIAGONAL IN L! k1, k2:", k1, k2)
                            # print(system.index_unflattener(k1), system.index_unflattener(k2))
                            hamiltonian_diag_l = False
                    
                    elif twoj1 != twoj2:
                        if not np.allclose(rho[k1, k2], 0):
                            rho_diag_lj = False
                        if not np.allclose(D[k1, k2], 0):
                            D_diag_lj = False
                        if not np.allclose(hamiltonian[k1, k2], 0):
                            hamiltonian_diag_lj = False
                    
                    elif twom1 != twom2:
                        if not np.allclose(rho[k1, k2], 0):
                            rho_diag_ljm = False
                        if not np.allclose(D[k1, k2], 0):
                            D_diag_ljm = False
                        if not np.allclose(hamiltonian[k1, k2], 0):
                            hamiltonian_diag_ljm = False
                    
                    # if twoj1 != twoj2:
                    #     print(rho[k1, k2] == 0)

            print("RHO diagonal in l:", rho_diag_l)
            print("RHO diagonal in l, j:", rho_diag_lj)
            print("RHO diagonal in l, j, m:", rho_diag_ljm)
            print("D diagonal in l:", D_diag_l)
            print("D diagonal in l, j:", D_diag_lj)
            print("D diagonal in l, j, m:", D_diag_ljm)
            print("Hamiltonian diagonal in l:", hamiltonian_diag_l)
            print("Hamiltonian diagonal in l, j:", hamiltonian_diag_lj)
            print("Hamiltonian diagonal in l, j, m:", hamiltonian_diag_ljm)


            # print(D)
            print("RHO\n", rho)
            print(hamiltonian)

            #print(D[0:8, 0:8])
            # print(rho[-8:, -8:])

            if not D_diag_l:
                exit()

            # Is D unitary?
            print("D unitary:", np.allclose(np.dot(D, D.conj().T), np.eye(self.num_states)))
            # Is the hamiltonian hermitian?
            print("H hermitian:", np.allclose(hamiltonian, hamiltonian.conj().T))

            # Diagonalize the single-particle hamiltonian
            sp_energies, eigenvectors = eigh(hamiltonian)

            #print("spe1", np.dot(hamiltonian, eigenvectors[:, 0]), eigenvectors[:, 0])

            # hr = hamiltonian[::2, ::2]
            # print(hamiltonian)
            # print(hr)
            # print(eigh(hr)[1])
            # Order the eigenvectors by energy (I think this is not necessary, but it's good to be sure)
            #eigenvectors = eigenvectors[:, np.argsort(sp_energies)]

            # Stop iterating if difference between previous energies is smaller than tolerance
            norm = np.linalg.norm(sp_energies - sp_energies_prev) / self.num_states
            if norm < tolerance:
                print("")
                print("~~~~~~~~~~~~~~~~~~~~~~ FULL SOLVER ~~~~~~~~~~~~~~~~~~~~~~~~")
                print("----------------------------------------------")
                print("Convergence reached after {} iterations".format(i + 1))
                print("Single particle energies: {}".format(sp_energies))
                print("NORM", norm)
                hf_energy = self.compute_hf_energy(sp_energies, t_matrix, V_matrix, V_non_asym, rho, D, self.num_particles)
                return hf_energy, sp_energies, eigenvectors
            
            # Update the previous energies and the D matrix
            sp_energies_prev = deepcopy(sp_energies)
            D = eigenvectors * (1 - mixture) + prev_D * mixture
            prev_D = deepcopy(D)

            print(sp_energies)

        # If we get here, we have reached the maximum number of iterations
        print("No convergence reached after {} iterations".format(max_iterations))
        return -1 


    def get_density_matrix(self, D):
        # Each element of the density matrix can be calculated as:
        # rho_alpha,beta = sum_i D_alpha,i * D*_beta,i = sum_i D_alpha,i * D^dagger_i,beta 
        # (where i runs over the hole states, i.e., up to the number of particles)
        # So we can write:

        trunc_D = D[:, :self.num_particles]
        trunc_D_dagger = trunc_D.conj().T
        rho = np.dot(trunc_D, trunc_D_dagger)

        return rho

    # Return the single particle hamiltonian
    
    def get_hamiltonian(self, t_matrix, V_matrix, rho):
        # There is a conflict between the notes and the PDF -> not sure if contraction is over indices 2, 4 or 4, 2...
        # Gamma is a contraction over mu, sigma of: V_alpha,sigma,beta,mu * rho_mu,sigma
        gamma = np.einsum('abcd,db->ac', V_matrix, rho)

        # for k1, k2 in product(range(t_matrix.shape[0]), range(t_matrix.shape[1])):
        #     print(k1, k2)
        #     print(self.system.index_unflattener(k1), self.system.index_unflattener(k2))
        #     print(V_matrix[k1, :, k2, :])

        # Are t and gamma hermitian?
        print("Gamma hermitian:", np.allclose(gamma, gamma.conj().T))
        print("Gamma all zeros?:", np.allclose(gamma, 0))
        
        print("GAMMA")
        print(gamma)

        #print(V_matrix[0,0,:,:])

        return t_matrix + gamma

    @staticmethod
    def compute_hf_energy(sp_energies, t_matrix, V_matrix, V_non_asym, rho, D, n_particles):

        # Change the basis of the matrices to the HF basis
        D_dagger = D.conj().T

        rho_hf = np.dot(D_dagger, np.dot(rho, D))
        t_matrix_hf = np.dot(D_dagger, np.dot(t_matrix, D))

        start = time.time()
        V_matrix_prov = np.einsum('abcd,ea,df->ebcf', V_matrix, D_dagger, D, optimize='optimal')
        V_matrix_hf = np.einsum('ebcf,gb,ch->eghf', V_matrix_prov, D_dagger, D, optimize='optimal')

        V_non_asym_prov = np.einsum('abcd,ea,df->ebcf', V_non_asym, D_dagger, D, optimize='optimal')
        V_non_asym_hf = np.einsum('ebcf,gb,ch->eghf', V_non_asym_prov, D_dagger, D, optimize='optimal')
        print("Time to change basis (V):", time.time() - start)

        #testV = hf_numba_helpers.n_change_basis_4d(V_matrix, D)
 
        # For comparison purposes:
        rho_trunc = rho_hf[:n_particles, :n_particles]
        V_trunc = V_matrix_hf[:n_particles, :n_particles, :n_particles, :n_particles]
        V_non_asym_trunc = V_non_asym_hf[:n_particles, :n_particles, :n_particles, :n_particles]
        t_trunc = t_matrix_hf[:n_particles, :n_particles]
        
        hf_energy = 0.5 * (np.trace(t_trunc) + np.sum(sp_energies[:n_particles]))
        
        # print("HF ENERGY", hf_energy)
        # exit()
        # return hf_energy

        # These two are wrong (has to be with the non-asym potential)
        e_hartree = 0.5 * np.einsum('ijkl,ki,lj', V_non_asym_trunc, rho_trunc, rho_trunc, optimize='optimal')
        e_fock = -0.5 * np.einsum('ijlk,ki,lj', V_non_asym_trunc, rho_trunc, rho_trunc, optimize='optimal')
        hf_energy_2 = np.sum(sp_energies[:n_particles]) - e_hartree - e_fock

        e_kin = np.einsum('ij,ji', t_trunc, rho_trunc)
        e_int = 0.5 * np.einsum('ijkl,ki,lj', V_trunc, rho_trunc, rho_trunc, optimize='optimal')

        # Another way of getting the energy
        hf_energy_3 = np.trace(t_trunc) + 0.5 * np.einsum('ijij', V_trunc)
        hf_energy_4 = np.sum(sp_energies[:n_particles]) - 0.5 * np.einsum('ijij', V_trunc)
        #hf_energy_5 = np.einsum('ab,ba', t_trunc, rho_trunc) + 0.5 * np.einsum('abcd,db,ca', V_trunc, rho_trunc, rho_trunc)
        hf_energy_5 = np.einsum('ab,ba', t_matrix, rho) + 0.5 * np.einsum('abcd,db,ca', V_matrix, rho, rho, optimize='optimal')
        #print(np.trace(t_trunc), 0.5 * np.trace(np.einsum('ijij', V_trunc)))

        # print(t_trunc)
        # print(t_matrix)
    
        print("SDAS", hf_energy, hf_energy_2, hf_energy_3, hf_energy_4, hf_energy_5)
        print("E_hartree", e_hartree)
        print("E_fock", e_fock)
        print("E_kin (not really)", e_kin)
        print("E_int", e_int)

        return hf_energy
    


if __name__ == "__main__":

    system = hfs.System(Ne_max=2, l_max=1, hbar_omega=10, mass=939)
    print("asdfsa")
    solver = Solver(system, num_particles=2)

    print("Number of states:", system.num_states)

    start_time = time.time()
    hf_energy, sp_energies, eigenvectors = solver.run(max_iterations=300, mixture=0)

    end_time = time.time()
    print("Time elapsed: {} seconds".format(end_time - start_time))

    print("Hartree-Fock energy: {} MeV".format(hf_energy))

    print(system.num_states)

    # wf = lambda r: h3d.wavefunction(r, k=1, l=0, hbar_omega=1, mass=1)

    # r = np.linspace(0, 10, 1000)

    # plt.plot(r, wf(r))
    # plt.show()