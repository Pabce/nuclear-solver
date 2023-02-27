import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from copy import deepcopy
import time
np.set_printoptions(precision=3, suppress=True, linewidth=200)

import minnesota_hf_system as hfs
import harmonic_3d as h3d



class Solver:

    def __init__(self, system, n_particles=2):
        self.n_particles = n_particles

        # Class containing the model information
        self.system = system

        self.num_states = system.num_states

        # Pick the hole states as those with the lowest energies
        self.original_energies = system.eigenenergies
        self.hole_states = np.sort(system.energy_indices[:self.n_particles])
        self.particle_states = np.sort(system.energy_indices[self.n_particles:])

        print("Hole states: {}".format(self.hole_states))
        print("Particle states: {}".format(self.particle_states))

    def run(self, max_iterations=100, tolerance=1e-8, mixture=0):
        # Hartee-Fock single-particle hamiltonian
        h = np.zeros((self.num_states, self.num_states))

        # Single-particle energies
        sp_energies = np.sort(self.original_energies)
        sp_energies_prev = np.zeros((self.num_states, 1))

        # D (overlap) matrix (transformation between original basis and HF basis, components D_alpha,q) (initial guess)
        D = np.eye(self.num_states)
        prev_eigenvectors = np.eye(self.num_states)

        # Fisrt, get 1-body and 2-body matrix elements in the original basis (only need to do this once!)
        t_matrix = self.system.get_one_body_matrix_elements()
        V_matrix = self.system.get_two_body_matrix_elements()
        #V_matrix = np.zeros((self.num_states, self.num_states, self.num_states, self.num_states))

        # print(V_matrix[0,1,:,:])
        # print(V_matrix[0,0,:,:])

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
                    print(V_matrix[n1,n2,:,:] - V_matrix[n1,n2,:,:].T)
                    print(V_matrix[n1,n2,:,:].T)
                    all_V_antisym = False
        
        print("V antisymmetric in the exchange of last two indices:", all_V_antisym)

        # TODO: We want to reorder our matrices so that the hole states are first
        print(sp_energies)
        for i in range(max_iterations):
            print("")
            print("----------------------------------------------")
            print("Iteration", i + 1)

            # Construct the density matrix in the original basis (in the first iteration, it will be non-zero only for the hole states)
            rho = self.get_density_matrix(D)
            #print(rho)

            # Construct the single-particle hamiltonian in the original basis
            hamiltonian = self.get_hamiltonian(t_matrix, V_matrix, rho)
            #print(hamiltonian)

            # TESTS:
            # Is rho equal to its sqaure?
            print("RHO equal to RHO^2:", np.allclose(rho, np.dot(rho, rho)))
            # Is rho hermitian?
            print("RHO hermitian:", np.allclose(rho, rho.conj().T))
            # Is rho diagonal in spin?
            print("RHO diagonal in spin:", np.allclose(rho[0::2, 1::2], 0, atol=1e-10))
            # print(rho)
            # print(rho[0::2, 1::2])
            #print(hamiltonian)

            # Is D unitary?
            print("D unitary:", np.allclose(np.dot(D, D.conj().T), np.eye(self.num_states)))
            # Is the hamiltonian hermitian?
            print("H hermitian:", np.allclose(hamiltonian, hamiltonian.conj().T))
            # # Is the hamiltonian diagonal in spin?
            # print("H diagonal in spin:", np.all(hamiltonian[0::2, 1::2]) == 0)


            # Diagonalize the single-particle hamiltonian
            sp_energies, eigenvectors = eigh(hamiltonian) #, subset_by_index=[0, self.num_states - 1])
            # Order the eigenvectors by energy (I think this is not necessary, but it's good to be sure)
            eigenvectors = eigenvectors[:, np.argsort(sp_energies)]
            sp_energies = np.sort(sp_energies)

            # Stop iterating if difference between previous energies is smaller than tolerance
            norm = np.linalg.norm(sp_energies - sp_energies_prev) / self.num_states
            if norm < tolerance:
                print("")
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print("----------------------------------------------")
                print("Convergence reached after {} iterations".format(i + 1))
                print("NORM", norm)
                print("Single particle energies: {}".format(sp_energies))

                hf_energy = self.compute_hf_energy(sp_energies, t_matrix, V_matrix, rho, D, self.n_particles)
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
        # There is a conflict between the notes and the PDF -> not sure if contraction is over indices 2, 4 or 4, 2...
        # Gamma is a contraction over mu, sigma of: V_alpha,sigma,beta,mu * rho_mu,sigma
        gamma = np.einsum('abcd,db->ac', V_matrix, rho)

        # Is gamma diagonal in spin?
        print("Gamma diagonal in spin:", np.allclose(gamma[0::2, 1::2], 0))

        print(gamma)

        return t_matrix + gamma

    @staticmethod
    def compute_hf_energy(sp_energies, t_matrix, V_matrix, rho, D, n_particles):

        # Change the basis of the matrices to the HF basis
        D_dagger = D.conj().T

        rho_hf = np.dot(D_dagger, np.dot(rho, D))
        t_matrix_hf = np.dot(D_dagger, np.dot(t_matrix, D))

        V_matrix_prov = np.einsum('abcd,ea,df->ebcf', V_matrix, D_dagger, D)
        V_matrix_hf = np.einsum('ebcf,gb,ch->eghf', V_matrix_prov, D_dagger, D)

 
        # For comparison purposes:
        rho_trunc = rho_hf[:n_particles, :n_particles]
        V_trunc = V_matrix_hf[:n_particles, :n_particles, :n_particles, :n_particles]
        t_trunc = t_matrix_hf[:n_particles, :n_particles]
        
        hf_energy = 0.5 * (np.trace(t_trunc) + np.sum(sp_energies[:n_particles]))

        # These two are wrong (has to be with the non-asym potential)
        e_hartree = 0.5 * np.einsum('ijkl,ki,lj', V_trunc, rho_trunc, rho_trunc)
        e_fock = -0.5 * np.einsum('ijlk,ki,lj', V_trunc, rho_trunc, rho_trunc)
        hf_energy_2 = np.sum(sp_energies[:n_particles]) - e_hartree - e_fock

        e_kin = np.einsum('ij,ji', t_trunc, rho_trunc)
        e_int = 0.5 * np.einsum('ijkl,ki,lj', V_trunc, rho_trunc, rho_trunc)

        # Another way of getting the energy
        hf_energy_3 = np.trace(t_trunc) + 0.5 * np.einsum('ijij', V_trunc)
        hf_energy_4 = np.sum(sp_energies[:n_particles]) - 0.5 * np.einsum('ijij', V_trunc)
        #hf_energy_5 = np.einsum('ab,ba', t_trunc, rho_trunc) + 0.5 * np.einsum('abcd,db,ca', V_trunc, rho_trunc, rho_trunc)
        hf_energy_5 = np.einsum('ab,ba', t_matrix, rho) + 0.5 * np.einsum('abcd,db,ca', V_matrix, rho, rho)
        #print(np.trace(t_trunc), 0.5 * np.trace(np.einsum('ijij', V_trunc)))

        # print(t_trunc)
        # print(t_matrix)
    
        print("SDAS", hf_energy, hf_energy_3, hf_energy_4, hf_energy_5)
        # print("E_hartree", e_hartree)
        # print("E_fock", e_fock)
        print("E_kin (not really)", e_kin)
        print("E_int", e_int)

        return hf_energy


if __name__ == "__main__":

    # Mass is neutron mass, you fucking retard
    system = hfs.System(k_num=5, l_num=1, hbar_omega=10, mass=939)
    solver = Solver(system, n_particles=8)

    start_time = time.time()
    hf_energy, sp_energies, eigenvectors = solver.run(max_iterations=2000, mixture=0.0)
    end_time = time.time()
    print("Time elapsed: {} seconds".format(end_time - start_time))

    print("Hartree-Fock energy: {} MeV".format(hf_energy))

    # wf = lambda r: h3d.wavefunction(r, k=1, l=0, omega=1, mass=1)

    # r = np.linspace(0, 10, 1000)

    # plt.plot(r, wf(r))
    # plt.show()