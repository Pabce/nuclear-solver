import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from copy import deepcopy
import time
np.set_printoptions(precision=2)

import full_minnesota_hf_system as hfs
import harmonic_3d as h3d


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
        t_matrix = self.system.matrix_ndflatten(self.system.get_one_body_matrix_elements(), dim=2)
        V_matrix = self.system.matrix_ndflatten(self.system.get_two_body_matrix_elements(), dim=4)

        print(V_matrix[0,1,:,:])

        # TESTS:
        # Is t_matrix hermitian?
        print("T hermitian:", np.allclose(t_matrix, t_matrix.conj().T))

        # Is V_matrix symmetric under swaping (1,2)<->(3,4)?
        all_V_sym = True
        for n1 in range(self.num_states):
            for n2 in range(self.num_states):
                herm = np.allclose(V_matrix[n1,n2,:,:], V_matrix[:,:,n1,n2], atol=1e-8)

                if not herm:
                    print("NOT SYMMETRIC UNDER (1,2)<->(3,4)! n1, n2:", n1, n2)
                    print(V_matrix[n1,n2,:,:] - V_matrix[:,:,n1,n2])
                    #print(V_matrix[:,:,n1,n2])
                    all_V_sym = False

        print("V symmetric under (1,2)<->(3,4):", all_V_sym)

        # Is V_matrix antisymmetric in the last two indices?
        all_V_antisym = True
        for n1 in range(self.num_states):
            for n2 in range(self.num_states):
                antisym = np.allclose(V_matrix[n1,n2,:,:], -V_matrix[n1,n2,:,:].T, atol=1e-8)

                if not antisym:
                    print("NOT ASYM! n1, n2:", n1, n2)
                    #print(V_matrix[n1,n2,:,:] + V_matrix[n1,n2,:,:].T)
                    # print(V_matrix[n1,n2,:,:] - V_matrix[n1,n2,:,:].T)
                    # print(V_matrix[n1,n2,:,:].T)
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
            # Is rho diagonal in m? (should it be???) (In any case, fix this for l!=0)
            print("RHO diagonal in m:", np.allclose(rho[0::2, 1::2], 0, atol=1e-10))
            # Is D unitary?
            print("D unitary:", np.allclose(np.dot(D, D.conj().T), np.eye(self.num_states)))
            # Is the hamiltonian hermitian?
            print("H hermitian:", np.allclose(hamiltonian, hamiltonian.conj().T))

            # Print...
            # print("HAMILTONIAN")
            # print(np.matrix(hamiltonian))


            # Diagonalize the single-particle hamiltonian
            sp_energies, eigenvectors = eigh(hamiltonian)
            # Order the eigenvectors by energy (I think this is not necessary, but it's good to be sure)
            eigenvectors = eigenvectors[:, np.argsort(sp_energies)]

            # Stop iterating if difference between previous energies is smaller than tolerance
            norm = np.linalg.norm(sp_energies - sp_energies_prev) / self.num_states
            if norm < tolerance:
                print("Convergence reached after {} iterations".format(i + 1))
                print("Single particle energies: {}".format(sp_energies))
                print("NORM", norm)
                hf_energy = self.compute_hf_energy(sp_energies, t_matrix, self.num_particles)
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
    @staticmethod
    def get_hamiltonian(t_matrix, V_matrix, rho):
        # There is a conflict between the notes and the PDF -> not sure if contraction is over indices 2, 4 or 4, 2...
        # Gamma is a contraction over mu, sigma of: V_alpha,sigma,beta,mu * rho_mu,sigma
        gamma = np.einsum('abcd,bd->ac', V_matrix, rho)

        # Are t and gamma hermitian?
        print("Gamma hermitian:", np.allclose(gamma, gamma.conj().T))
        # print("GAMMA")
        # print(gamma)

        #print(V_matrix[0,0,:,:])

        return t_matrix + gamma

    @staticmethod
    def compute_hf_energy(sp_energies, t_matrix, num_particles):
        return 0.5 * (np.trace(t_matrix[:num_particles]) + np.sum(sp_energies[:num_particles]))
    


if __name__ == "__main__":

    system = hfs.System(Ne_max=8, l_max=0, hbar_omega=3, mass=939)
    solver = Solver(system, num_particles=8)

    print("Number of states:", system.num_states)

    start_time = time.time()
    hf_energy, sp_energies, eigenvectors = solver.run(max_iterations=3, mixture=0)

    total_e, t_e, sp_e = hf_energy

    end_time = time.time()
    print("Time elapsed: {} seconds".format(end_time - start_time))

    print("Hartree-Fock energy: {} MeV".format(total_e))
    print("T component: {} MeV".format(t_e))
    print("SP component: {} MeV".format(sp_e))

    print(system.num_states)

    # wf = lambda r: h3d.wavefunction(r, k=1, l=0, hbar_omega=1, mass=1)

    # r = np.linspace(0, 10, 1000)

    # plt.plot(r, wf(r))
    # plt.show()