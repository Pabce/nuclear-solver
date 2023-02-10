import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from copy import deepcopy
import time
np.set_printoptions(precision=3)

import full_minnesota_hf_system as hfs
import harmonic_3d as h3d

# TODO: Alternative to this class that flattens the matrices, would be much cleaner, I think

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


    def run(self, max_iterations=1000, tolerance=1e-8, mixture=0):
        # Hartee-Fock single-particle hamiltonian: we have two indices for the n quantum number, and one index for both the l and j quantum numbers
        # (as HF single particle Hamiltonian is diagonal in the l and j quantum numbers)
        # Also, only HO states with same lj quantum numbers as HF states contribute
        # n1, n2, l, twoj (using the convetion for twoj to be 0 or 1...)

        h = np.zeros((self.n_level_max + 1, self.n_level_max + 1, self.l_level_max + 1, 2))

        # Single-particle energies
        sp_energies_flat = self.original_energies_flat
        sp_energies_prev = np.zeros((self.num_states, 1))
        quantum_numbers = self.original_quantum_numbers

        # D (overlap) matrix (transformation between original basis and HF basis, components D_alpha,q) (initial guess)
        # Same structure as h (1 if n1 == n2, indep of l, j, m, values, 0 otherwise)
        D = np.zeros((self.n_level_max + 1, self.n_level_max + 1, self.l_level_max + 1, 2))
        for n1 in range(self.n_level_max + 1):
            D[n1, n1, :, :] = 1

        prev_D = deepcopy(D)

        # First, get 1-body and 2-body matrix elements in the original basis (only need to do this once!)
        t_matrix = self.format_t_matrix(self.system.get_one_body_matrix_elements())
        V_matrix = self.system.get_two_body_matrix_elements()

        # TODO: We want to reorder our matrices so that the hole states are first (maybe we don't actually need this)

        print("ORIGINAL HO ENERGIES:", sp_energies_flat)
        print("HO QUANTUM NUMBERS:", quantum_numbers)
        for i in range(max_iterations):
            #print(D[:,:,0,1])
            print("---------------------")
            print("Iteration", i + 1)

            # Build the occupation matrix
            occ_matrix = self.get_occupation_matrix(sp_energies_flat, quantum_numbers)
            # Print the sum of all elements in the occupation matrix (should be equal to the number of particles)
            #print("Occupation matrix sum: {}".format(np.sum(occ_matrix)))

            # Construct the density matrix in the original basis (in the first iteration, it will be non-zero only for the hole states)
            rho = self.get_density_matrix(D, occ_matrix)

            # print(D[:,:,0,1])
            print(occ_matrix[:,0,1])
            print(np.dot(D[:,:,0,1], D[:,:,0,1].T))
            
            #print(np.dot(rho[:,:,0,1], rho[:,:,0,1]) - rho[:,:,0,1])
            #print((np.einsum('abcd, bxcd -> axcd', rho, rho) - rho)[:,:,0,1])

            # Print rho up to 3 decimals
            #print(rho)
            # print(t_matrix)
            # print(V_matrix)

            # Construct the single-particle hamiltonian in the original basis
            hamiltonian = self.get_hamiltonian(t_matrix, V_matrix, rho)
            #print(hamiltonian)

            # Diagonalize the single-particle hamiltonian
            sp_energies, sp_energies_flat, eigenvectors, quantum_numbers = self.diagonalize_hamiltonian(hamiltonian)

            # Stop iterating if difference between previous energies is smaller than tolerance
            norm = np.linalg.norm(sp_energies_flat - sp_energies_prev) / self.num_states
            print("NORM", norm, self.num_states)
            if norm < tolerance:
                print("Convergence reached after {} iterations".format(i + 1))
                print("Single particle energies: {}".format(sp_energies_flat))
                print("NORM", norm)
                hf_energy = self.compute_hf_energy(sp_energies_flat, t_matrix, self.num_particlesç, occ_matrix)
                return hf_energy, sp_energies, eigenvectors
            
            # Update the previous energies and the D matrix (properly)
            sp_energies_prev = deepcopy(sp_energies_flat)
            D = self.update_D(eigenvectors, prev_D, mixture)
            prev_D = deepcopy(D)

            print(sp_energies_flat)

        # If we get here, we have reached the maximum number of iterations
        print("No convergence reached after {} iterations".format(max_iterations))
        return -1 


    def format_t_matrix(self, t_matrix):
        # We need to index the t matrix in the same way as the gamma matrix
        format_t_matrix = np.zeros((self.n_level_max + 1, self.n_level_max + 1, self.l_level_max + 1, 2))

        for l in range(self.l_level_max + 1):
            format_t_matrix[:, :, l, 0] = t_matrix[:, l, 0, :, l, 0]
            format_t_matrix[:, :, l, 1] = t_matrix[:, l, 1, :, l, 1]

        return format_t_matrix


    def get_density_matrix(self, D, occupation_matrix):
        # Each element of the density matrix can be calculated as:
        # rho_n1,n2^l,j = SUM over nbarp: O^nbarp,l,j * D_n1,nbarp^l,j * D*_n2,nbarp^l,j

        rho = np.zeros((self.n_level_max + 1, self.n_level_max + 1, self.l_level_max + 1, 2))
        D_star = D.conj()

        # Build the density matrix
        for l in range(self.l_level_max + 1):
            for twoj in range(np.abs(2*l - 1), 2*l + 2, 2):
                twoj_idx = 0 if twoj < 2*l else 1

                rho[:, :, l, twoj_idx] = np.einsum('an,bn,n->ab', D[:, :, l, twoj_idx], D_star[:, :, l, twoj_idx], occupation_matrix[:, l, twoj_idx])

        return rho


    # Return the single particle hamiltonian
    @staticmethod
    def get_hamiltonian(t_matrix, V_matrix, rho):
        # Gamma_(n1,n3)^(l,j) = contraction over n4, n2, l', j' of: V_(n1 l j),(n2 l' j'),(n3 l j),(n4 l' j') * rho_(n4, n2)^(l' j')

        gamma = np.einsum('abc def gbc ief, idef -> agbc', V_matrix, rho)

        return t_matrix + gamma
    

    # TODO: try the "flatten" approach and see if it changes anything
    # (you will have to do this in the future anyways, as your HF states won't necessarily share the ljm quantum numbers with the HO basis)
    def diagonalize_hamiltonian(self, hamiltonian):
        # The diagonalization problem can be written as:
        # SUM in n3 of: h_(n1, n3)^l,j * D_(n3, nbar)^l,j = eps_(nbar,l,l) * D_(n1, nbar)^l,j

        eigenvectors = np.zeros((self.l_level_max + 1, 2), dtype=object)
        sp_energies = np.zeros((self.l_level_max + 1, 2), dtype=object)
        sp_energies_flat = []
        quantum_numbers = []

        for l in range(self.l_level_max + 1):
            for twoj in range(np.abs(2*l - 1), 2*l + 2, 2):
                twoj_idx = 0 if twoj < 2*l else 1

                h_lj = hamiltonian[:, :, l, twoj_idx]

                # Diagonalize the single-particle hamiltonian for a given l, j
                sp_energies_lj, eigenvectors_lj = eigh(h_lj)

                # We need to account for the m-degeneracy: 2j + 1 states/energies for each j
                # (here we don't need to use the computed occupation matrix, we are computing the single-particle energies 
                # independently of the total number of particles)

                sp_energies[l, twoj_idx] = np.repeat(sp_energies_lj, twoj + 1)
                sp_energies_flat.extend([_ for _ in sp_energies[l, twoj_idx]])
                eigenvectors[l, twoj_idx] = eigenvectors_lj
                for i in range(len(sp_energies_lj)):
                    quantum_numbers.extend([(i, l, twoj)] * (twoj + 1))
        
        # # Order the everything by energy (I think this is not necessary, but it's good to be sure)
        # sp_energies = np.array(sp_energies)
        # eigenvectors = np.array(eigenvectors)
        # quantum_numbers = np.array(quantum_numbers)
        # sp_energies = sp_energies[np.argsort(sp_energies)]
        # quantum_numbers = quantum_numbers[np.argsort(sp_energies)]
        # eigenvectors = eigenvectors[:, np.argsort(sp_energies)]

        sp_energies_flat = np.sort(sp_energies_flat)

        # print("SDDDDDadfs")
        # print(hamiltonian[:,:,0,1])
        # print(sp_energies[0,1])
        
        
        return sp_energies, sp_energies_flat, eigenvectors, quantum_numbers

    
    def update_D(self, eigenvectors, mixture, prev_D):
        # print(eigenvectors[0,0])
        # print("SDSDAS")

        new_D = np.zeros((self.n_level_max + 1, self.n_level_max + 1, self.l_level_max + 1, 2))
        
        for l in range(self.l_level_max + 1):
            for twoj in range(np.abs(2*l - 1), 2*l + 2, 2):
                twoj_idx = 0 if twoj < 2*l else 1

                new_D[:, :, l, twoj_idx] = eigenvectors[l, twoj_idx]

        new_D = new_D * (1 - mixture) + prev_D * mixture

        return new_D


    def compute_hf_energy(self, sp_energies, t_matrix, num_particles, occupation_matrix):
        t_component = 0

        for l in range(self.l_level_max + 1):
            for twoj in range(np.abs(2*l - 1), 2*l + 2, 2):
                twoj_idx = 0 if twoj < 2*l else 1

                for n in range(self.n_level_max + 1):
                    t_component += t_matrix[n, n, l, twoj_idx] * occupation_matrix[n, l, twoj_idx]

        sp_component = 0.5 * np.sum(sp_energies[:num_particles])
        t_component *= 0.5

        total_energy = t_component + sp_component


        return (total_energy, t_component, sp_component)


    # Let's do this the more general way, without assuming that only states with the same lj contribute:
    # Fill up states in order of increasing energy, until we reach the desired number of particles
    def get_occupation_matrix(self, sp_energies_flat, quantum_numbers):
        total_occupation = 0
        occupation_matrix = np.zeros((self.n_level_max + 1, self.l_level_max + 1, 2))
        
        for i, energy in enumerate(sp_energies_flat):
            n, l, twoj = quantum_numbers[i]
            twoj_idx = 0 if twoj < 2*l else 1

            occupation_num = 1 # We are explicitly iterating over all 2j + 1 states for each j, so we can just set this to 1
            total_occupation += occupation_num

            if total_occupation > self.num_particles:
                occupation_num -= total_occupation - self.num_particles
                total_occupation = self.num_particles

            occupation_matrix[n, l, twoj_idx] += occupation_num
        
        occupation_matrix[occupation_matrix > 1] = 1

        return occupation_matrix


if __name__ == "__main__":

    system = hfs.System(Ne_max=8, l_max=0, hbar_omega=3, mass=1)
    solver = Solver(system, num_particles=8)

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