import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from copy import deepcopy
import time

import minnesota_hf_system as hfs
import harmonic_3d as h3d



class Solver:

    def __init__(self, system, n_particles=2):
        self.n_particles = n_particles

        # Class containing the model information
        self.system = system

        self.n_states = system.n_states

        # Pick the hole states as those with the lowest energies
        self.original_energies = system.eigenenergies
        self.hole_states = np.sort(system.energy_indices[:self.n_particles])
        self.particle_states = np.sort(system.energy_indices[self.n_particles:])

        print("Hole states: {}".format(self.hole_states))
        print("Particle states: {}".format(self.particle_states))

    def run(self, max_iterations=100, tolerance=1e-8, mixture=0):
        # Hartee-Fock single-particle hamiltonian
        h = np.zeros((self.n_states, self.n_states))

        # Single-particle energies
        sp_energies = np.sort(self.original_energies)
        sp_energies_prev = np.zeros((self.n_states, 1))

        # D (overlap) matrix (transformation between original basis and HF basis, components D_alpha,q) (initial guess)
        D = np.eye(self.n_states)
        prev_eigenvectors = np.eye(self.n_states)

        # Fisrt, get 1-body and 2-body matrix elements in the original basis (only need to do this once!)
        t_matrix = self.system.get_one_body_matrix_elements()
        V_matrix = self.system.get_two_body_matrix_elements()

        # TODO: We want to reorder our matrices so that the hole states are first

        print(sp_energies)
        for i in range(max_iterations):
            print("Iteration", i + 1)

            # Construct the density matrix in the original basis (in the first iteration, it will be non-zero only for the hole states)
            rho = self.get_density_matrix(D)

            # Print rho up to 3 decimals
            np.set_printoptions(precision=3)
            #print(rho)
            # print(t_matrix)
            # print(V_matrix)

            # Construct the single-particle hamiltonian in the original basis
            hamiltonian = self.get_hamiltonian(t_matrix, V_matrix, rho)
            #print(hamiltonian)

            # Diagonalize the single-particle hamiltonian
            sp_energies, eigenvectors = eigh(hamiltonian, subset_by_index=[0, self.n_states - 1])
            # Order the eigenvectors by energy (I think this is not necessary, but it's good to be sure)
            eigenvectors = eigenvectors[:, np.argsort(sp_energies)]

            # Stop iterating if difference between previous energies is smaller than tolerance
            norm = np.linalg.norm(sp_energies - sp_energies_prev) / self.n_states
            if norm < tolerance:
                print("Convergence reached after {} iterations".format(i + 1))
                print("Single particle energies: {}".format(sp_energies))
                print("NORM", norm)
                hf_energy = self.compute_hf_energy(sp_energies, t_matrix, self.n_particles)
                return hf_energy, sp_energies, eigenvectors
            
            # Update the previous energies and the D matrix
            sp_energies_prev = deepcopy(sp_energies)
            D = eigenvectors * (1 - mixture) + prev_eigenvectors * mixture
            prev_eigenvectors = deepcopy(eigenvectors)

            print(sp_energies)


        # If we get here, we have reached the maximum number of iterations
        print("No convergence reached after {} iterations".format(max_iterations))
        return -1 


    def get_density_matrix(self, D):
        # Each element of the density matrix can be calculated as:
        # rho_alpha,beta = sum_i D_alpha,i * D*_beta,i = sum_i D_alpha,i * D^dagger_i,beta 
        # (where i runs over the hole states, i.e., up to the number of particles)
        # So we can write:

        trunc_D = D[:, :self.n_particles]
        trunc_D_dagger = trunc_D.conj().T
        rho = np.dot(trunc_D, trunc_D_dagger)

        return rho

    # Return the single particle hamiltonian
    @staticmethod
    def get_hamiltonian(t_matrix, V_matrix, rho):
        # Gamma is a contraction over mu, sigma of: V_mu,alpha,sigma,beta * rho_mu,sigma
        gamma = np.einsum('abcd,ac->bd', V_matrix, rho)

        return t_matrix + gamma

    @staticmethod
    def compute_hf_energy(sp_energies, t_matrix, n_particles):
        return 0.5 * (np.trace(t_matrix[:n_particles]) + np.sum(sp_energies[:n_particles]))


if __name__ == "__main__":

    system = hfs.System(k_num=14, l_num=1, omega=3, mass=1)
    solver = Solver(system, n_particles=8)

    start_time = time.time()
    hf_energy, sp_energies, eigenvectors = solver.run(mixture=0.0)
    end_time = time.time()
    print("Time elapsed: {} seconds".format(end_time - start_time))

    print("Hartree-Fock energy: {} MeV".format(hf_energy))

    # wf = lambda r: h3d.wavefunction(r, k=1, l=0, omega=1, mass=1)

    # r = np.linspace(0, 10, 1000)

    # plt.plot(r, wf(r))
    # plt.show()