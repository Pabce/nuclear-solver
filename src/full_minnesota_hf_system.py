'''
Attempt to implement Hartree-Fock as shown in https://wikihost.nscl.msu.edu/TalentDFT/doku.php?id=projects
We start with a simplified neutron drop system in the harmonic oscillator basis.
'''

import numpy as np
from itertools import product
from scipy.integrate import quad, dblquad
import harmonic_3d as h3d
import time
import moshinsky_way as mw
import hf_numba_helpers 

from copy import deepcopy

HBAR = 197.3269788 # MeV * fm
#HBAR = 1

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

    def __init__(self, Ne_max, l_max, hbar_omega=1, mass=1, include_m=True) -> None:
        self.Ne_max = Ne_max
        # l_max represents the maximum value of l (cannot be larger than Ne_max)
        if l_max > Ne_max:
            raise ValueError("l_max cannot be larger than Ne_max")

        # l can go from Ne to 0 in steps of 2
        true_num_states = 0
        num_j_states = 0
        for Ne in range(Ne_max + 1):
            max_l = min(Ne, l_max)
            for l in range(Ne, -1, -2):
                if l > max_l:
                    continue
                for twoj in range(np.abs(2*l - 1), 2*l + 2, 2):
                    true_num_states += twoj + 1 # for the j and m_j degeneracy
                    num_j_states += 1

        self.true_num_states = int(true_num_states)
        self.num_j_states = int(num_j_states)
        self.num_states = true_num_states if include_m else num_j_states

        self.n_level_max = Ne_max // 2
        self.l_level_max = l_max

        self.hbar_omega = hbar_omega
        self.mass = mass

        self.include_m = include_m

        # List of all energies and their indices
        self.eigenenergies, self.eigenenergies_flat, self.quantum_numbers = self.get_lowest_energies(include_m=self.include_m)

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
                _, sqrt_norm = h3d.series_wavefunction(r, k=n, l=l, hbar_omega=self.hbar_omega, mass=self.mass)

                wavefunctions[Ne, l] = lambda r, k=n, l=l: h3d.wavefunction(r, k=k, l=l, hbar_omega=self.hbar_omega, mass=self.mass)
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
        # These elements only depend on n and l, not on the spin.
        # Again, pretty sure the above is wrong and the elements should be diagonal in the HO basis.

        t = np.zeros((self.Ne_max // 2 + 1, self.l_level_max + 1, 2, self.Ne_max // 2 + 1, self.l_level_max + 1, 2), dtype=np.float64)

        # They are also diagonal in l
        # TODO: check if this is actually diagonal in spin
        for n1, n2 in product(range(self.Ne_max // 2 + 1), repeat=2):
            for l in range(self.l_level_max + 1):
                if n1 == n2:
                    val = 2 * n1 + l + 3/2

                # elif n1 == n2 - 1:  #--> this is nonsensical, I think
                #     val = np.sqrt(n2 * (n2 + l + 1/2))
                # elif n1 == n2 + 1:
                #     val = np.sqrt(n1 * (n1 + l + 1/2))

                else:
                    val = 0

                for twoj_idx in range(2):
                    t[n1, l, twoj_idx, n2, l, twoj_idx] = val

        t *= 0.5 * self.hbar_omega

        return t
    

    # Get the antisymmetrized two-body matrix elements in the HO basis
    # TODO: reduce computations exploiting hermiticity of the potential
    # TODO: make sure the potential is antisymmetric, idiot
    def get_two_body_matrix_elements(self):

        Ne_max, l_max, hbar_omega, mass = self.Ne_max, self.l_level_max, self.hbar_omega, self.mass
        integration_limit, integration_steps = 40, 5500

        # TODO: Add a way to pass the potential into the Moshinsky class
        # Set the radial wavefunctions
        wfs = mw.set_wavefunctions(Ne_max, l_max, hbar_omega, mass, integration_limit, integration_steps)

        # Set the Moshinsky brackets, reading from file
        start = time.time()
        moshinsky_brackets = mw.set_moshinsky_brackets(Ne_max, l_max)
        print("Time to set Moshinsky brackets:", time.time() - start)

        # Set the Wigner 9j symbols, reading from file
        start = time.time()
        wigner_9j_dict = mw.set_wigner_9js()
        print("Time to set Wigner 9j symbols:", time.time() - start)

        print('---------------------------------------------------------------')

        # Set the central potential (not antisymmetrized!) reduced matrix elements. You have to do...
        # Minnesota potential
        # 1. R component: VR * 1/2*(1 + Pr)
        # 2. s component: Vs * 1/2*(1 - Psigma) * 1/2*(1 + Pr)
        # 3... Ignoring for now

        cp_mats = []
        cp_J_coupling_basis_mats = []
        V_mats = []
        for params in [(self.V0R, self.kappaR, "none", "even"), (-self.V0s, self.kappas, "singlet", "even"), (-self.V0t, self.kappat, "triplet", "even")]: 
            V0, kappa, spin_projector, parity_projector = params

            start = time.time()
            central_potential_reduced_matrix = mw.set_central_potential_reduced_matrix(wfs, V0, kappa,
                                                        Ne_max, l_max, integration_limit, integration_steps)
            print("Time to set reduced matrix elements:", time.time() - start)

            # Set the central potential matrix in ls coupling basis
            start = time.time()
            central_potential_ls_coupling_basis_matrix = mw.set_central_potential_ls_coupling_basis_matrix(central_potential_reduced_matrix,
                                                                                                        moshinsky_brackets, Ne_max, l_max)
            print("Time to set ls coupling basis matrix:", time.time() - start)
            # for n1, n2 in product(range(self.n_level_max), repeat=2):
            #     mt = central_potential_ls_coupling_basis_matrix[n1,0, n2,0, :,0, :,0, 0]
            #     mt2 = central_potential_ls_coupling_basis_matrix[:,0, :,0, n1,0, n2,0, 0]
            #     #mt = central_potential_ls_coupling_basis_matrix[n1,0,n2,0,:,0,:,0,0]
            #     print("CHECK: Is it hermitian?:", np.allclose(mt, mt2))

            #     if not np.allclose(mt, mt2):
            #         print(mt)
            #         print(mt2)

            # exit()
            
            # Set the central potential matrix in J coupling basis
            start = time.time()
            central_potential_J_coupling_matrix = mw.set_central_potential_J_coupling_basis_matrix(central_potential_ls_coupling_basis_matrix, 
                                                                                wigner_9j_dict, Ne_max, l_max, spin_projector, parity_projector)
            print("Time to set J coupling basis matrix:", time.time() - start)
            cp_J_coupling_basis_mats.append(central_potential_J_coupling_matrix)
            
            # Get the central potential matrix, once and for all
            start = time.time()
            central_potential_matrix = mw.set_central_potential_matrix(central_potential_J_coupling_matrix, Ne_max, l_max)
            print("Time to set central potential matrix:", time.time() - start)

            cp_mats.append(central_potential_matrix)

            print("------------------------------------------------------------------------------")
        
        # cp_mats have components (n1, l1, twoj1, n2, l2, twoj2, n3, l3, twoj3, n4, l4, twoj4)
        v = cp_mats[0] + cp_mats[1] + cp_mats[2]
        v *= -1

        v_a = v - v.transpose((0,1,2, 3,4,5, 9,10,11, 6,7,8))

        V_mat = v_a
        V_non_asym = v

        # print(self.num_states)
        # print(V_mat.shape)

        # Some checks
        print("All zeros?", np.allclose(V_mat, 0))
        print("Fraction of non-zero elements?", np.count_nonzero(V_mat)/V_mat.size)

        print("Asym potential -- All antisym", np.allclose(v_a, -v_a.transpose((0,1,2, 3,4,5, 9,10,11, 6,7,8))))
        #print("Asym potential -- All antisym 2", np.allclose(v_a, -v_a.transpose((3,4,5, 0,1,2, 6,7,8, 9,10,11))))

        # Some checks on V_mat
        # Is it sym?
        print("Not antisymmetrized potetial -- All sym?", np.allclose(v, v.transpose((6,7,8, 9,10,11, 0,1,2, 3,4,5))))
        print("Asym potetial -- All sym?", np.allclose(V_mat, V_mat.transpose((6,7,8, 9,10,11, 0,1,2, 3,4,5))))
        
        print("------------------ªªªªª------------------")

        # print(V_mat[0,0,1,1,0,1,:,0,1,:,0,1])
        # print(self.index_flattener(0,0,1,-1))
        # print(self.index_flattener(1,0,1,-1))


        

        return V_mat, V_non_asym, cp_J_coupling_basis_mats


    def get_two_body_matrix_elements_2(self):
        Ne_max, l_max, hbar_omega, mass = self.Ne_max, self.l_level_max, self.hbar_omega, self.mass
        integration_limit, integration_steps = 40, 5500

        # TODO: Add a way to pass the potential into the Moshinsky class
        # Set the radial wavefunctions
        wfs = mw.set_wavefunctions(Ne_max, l_max, hbar_omega, mass, integration_limit, integration_steps)

        # Set the Moshinsky brackets, reading from file
        start = time.time()
        moshinsky_brackets = mw.set_moshinsky_brackets(Ne_max, l_max)
        print("Time to set Moshinsky brackets:", time.time() - start)

        # Set the Wigner 9j symbols, reading from file
        start = time.time()
        wigner_9j_dict = mw.set_wigner_9js()
        print("Time to set Wigner 9j symbols:", time.time() - start)

        print('---------------------------------------------------------------')

        # We'll try to antisymmetrize directly

        cp_mats = []
        cp_J_coupling_basis_mats = []
        V_mats = []
        for params in [(self.V0R, self.kappaR, "singlet", "even"), (-self.V0s, self.kappas, "singlet", "even")]: 
            V0, kappa, spin_projector, parity_projector = params

            start = time.time()
            central_potential_reduced_matrix = mw.set_central_potential_reduced_matrix(wfs, V0, kappa,
                                                        Ne_max, l_max, integration_limit, integration_steps)
            print("Time to set reduced matrix elements:", time.time() - start)

            # Set the central potential matrix in ls coupling basis
            start = time.time()
            central_potential_ls_coupling_basis_matrix = mw.set_central_potential_ls_coupling_basis_matrix(central_potential_reduced_matrix,
                                                                                                        moshinsky_brackets, Ne_max, l_max, parity_projector)
            print("Time to set ls coupling basis matrix:", time.time() - start)
            # for n1, n2 in product(range(self.n_level_max), repeat=2):
            #     mt = central_potential_ls_coupling_basis_matrix[n1,0, n2,0, :,0, :,0, 0]
            #     mt2 = central_potential_ls_coupling_basis_matrix[:,0, :,0, n1,0, n2,0, 0]
            #     #mt = central_potential_ls_coupling_basis_matrix[n1,0,n2,0,:,0,:,0,0]
            #     print("CHECK: Is it hermitian?:", np.allclose(mt, mt2))

            #     if not np.allclose(mt, mt2):
            #         print(mt)
            #         print(mt2)

            # exit()
            
            # Set the central potential matrix in J coupling basis
            start = time.time()
            central_potential_J_coupling_matrix = mw.set_central_potential_J_coupling_basis_matrix(central_potential_ls_coupling_basis_matrix, 
                                                                                wigner_9j_dict, Ne_max, l_max, spin_projector, parity_projector)
            print("Time to set J coupling basis matrix:", time.time() - start)
            cp_J_coupling_basis_mats.append(central_potential_J_coupling_matrix)
            
            # Get the central potential matrix, once and for all
            start = time.time()
            central_potential_matrix = mw.set_central_potential_matrix(central_potential_J_coupling_matrix, Ne_max, l_max)
            print("Time to set central potential matrix:", time.time() - start)

            cp_mats.append(central_potential_matrix)

            print("------------------------------------------------------------------------------")
        
        # cp_mats have components (n1, l1, twoj1, n2, l2, twoj2, n3, l3, twoj3, n4, l4, twoj4)
        v = cp_mats[0]# + cp_mats[1]
        v *= 1

        vd = v
        ve_pr = v # v.transpose((0,1,2, 3,4,5, 6,7,11, 9,10,8))

        V_mat = vd
        #V_mat = V_mat - V_mat.transpose((0,1,2, 3,4,5, 9,10,11, 6,7,8))
        V_non_asym = v

        # print(self.num_states)
        # print(V_mat.shape)

        # Some checks
        print("All zeros?", np.allclose(V_mat, 0))
        print("Fraction of non-zero elements?", np.count_nonzero(V_mat)/V_mat.size)

        asym = np.allclose(V_mat, -V_mat.transpose((0,1,2, 3,4,5, 9,10,11, 6,7,8)))
        print("Asym potential -- All antisym:", asym)
        print("Asym J coupling basis -- All antisym:", np.allclose(cp_J_coupling_basis_mats[0], -cp_J_coupling_basis_mats[0].transpose((0,1,2, 3,4,5, 9,10,11, 6,7,8, 12))))
        #print("Asym potential -- All antisym 2", np.allclose(v_a, -v_a.transpose((3,4,5, 0,1,2, 6,7,8, 9,10,11))))

        V_mat = cp_J_coupling_basis_mats[0]
        if not asym:
            for n1, n2, n3, n4 in product(range(self.n_level_max + 1 -1), repeat=4):
                for l1, l2, l3, l4 in product(range(self.l_level_max + 1 -1), repeat=4):
                    for twoj1_idx, twoj2_idx, twoj3_idx, twoj4_idx in product(range(2), repeat=4):

                        if not np.allclose(V_mat[n1,l1,twoj1_idx, n2,l2,twoj2_idx, n3,l3,twoj3_idx, n4,l4,twoj4_idx],
                            -V_mat[n1,l1,twoj1_idx, n2,l2,twoj2_idx, n4,l4, twoj4_idx, n3,l3,twoj3_idx]):

                            print(n1,l1,twoj1_idx, "|", n2,l2,twoj2_idx, "|", n3,l3,twoj3_idx, "|", n4,l4,twoj4_idx)
                            print(V_mat[n1,l1,twoj1_idx, n2,l2,twoj2_idx, n3,l3,twoj3_idx, n4,l4,twoj4_idx])
                            print(V_mat[n1,l1,twoj1_idx, n2,l2,twoj2_idx, n4,l4,twoj4_idx, n3,l3,twoj3_idx])
                            print('------------------------------')

        exit()

        # Some checks on V_mat
        # Is it sym?
        print("Not antisymmetrized potetial -- All sym?", np.allclose(v, v.transpose((6,7,8, 9,10,11, 0,1,2, 3,4,5))))
        print("Asym potetial -- All sym?", np.allclose(V_mat, V_mat.transpose((6,7,8, 9,10,11, 0,1,2, 3,4,5))))
        
        print("------------------ªªªªª------------------")

        # print(V_mat[0,0,1,1,0,1,:,0,1,:,0,1])
        # print(self.index_flattener(0,0,1,-1))
        # print(self.index_flattener(1,0,1,-1))


        

        return V_mat, V_non_asym, cp_J_coupling_basis_mats


    # Energies of the HO basis eigenstates
    def get_ho_energies(self, Ne=-1, n=-1, l=-1):
        if Ne == -1:
            return 0.5 * self.hbar_omega * (2 * n + l + 3/2)
        else:
            return 0.5 * self.hbar_omega * Ne

    # Get the indices of the lowest energy states
    def get_lowest_energies(self, include_m):
        energies_nl = np.zeros((self.n_level_max + 1, self.l_level_max + 1))
        energies_flat = []
        quantum_numbers = []

        for n in range(self.n_level_max + 1):
            for l in range(self.l_level_max + 1):
                e = self.get_ho_energies(n=n, l=l)
                energies_nl[n, l] = e

                for twoj in range(np.abs(2 * l - 1), 2 * l + 2, 2):
                    if include_m:
                        energies_flat.extend([e] * (twoj + 1))
                        quantum_numbers.extend([(n, l, twoj)] * (twoj + 1))
                    else:
                        energies_flat.append(e)
                        quantum_numbers.append((n, l, twoj))

        #idx = np.argsort(energies)[:num]
        return energies_nl, energies_flat, quantum_numbers
    
    # --------------------------------------------------------------------------------------------------------
    # Methods to be used by the solver class to convert between quantum numbers and flat indices
    def index_flattener(self, n, l, twoj, twom=None):
        
        if twoj > 2 * l + 1 or twoj < np.abs(2 * l - 1):
            raise ValueError("The value of twoj={} is not allowed for the given value of l={}".format(twoj, l))
        
        if twom:
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
                    if not twom:
                        if n_p == n and lp == l and twojp == twoj:
                            return idx
                        idx += 1
                    else:
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
                    #print(n_p, lp, twojp)
                    if not self.include_m:
                        if idxp == idx:
                            return n_p, lp, twojp
                        idxp += 1
                    else:
                        for twomp in range(-twojp, twojp + 1, 2):
                            if idxp == idx:
                                return n_p, lp, twojp, twomp
                            idxp += 1

        raise ValueError("The index provided is too large")
    

    # This is probably mega-slow for larger systems and will need to be numbad --> Indeed.
    # TODO: Does this need any extra info on how to treat the ms? Does adding m_diagonal even make sense?
    # Make sure the chipmunks don't eat the hamiltonian
    def matrix_ndflatten(self, matrix, dim=2, include_m=False, m_diagonal=False, asym=False):
        shape = self.num_j_states if not include_m else self.num_states
        shape_tuple = (shape,) * dim
        flat_matrix = np.zeros(shape_tuple)

        it = np.nditer(matrix, flags=['multi_index', 'refs_ok'])

        n, l, twoj_idx, twoj = np.zeros(dim, dtype=int), np.zeros(dim, dtype=int), np.zeros(dim, dtype=int), np.zeros(dim, dtype=int)
        idx = np.zeros(dim, dtype=int)

        for el in it:
            for i in range(dim):
                n[i], l[i], twoj_idx[i] = it.multi_index[i * 3 : (i + 1) * 3]
                twoj[i] = 2 * l[i] - 1 + twoj_idx[i] * 2

            # Special unphysical case: FIXME
            if np.any(twoj < 0):
                #print("Skipping", n, l, twoj)
                continue
            # Unneeded values for the flattened matrix 
            if np.any(2 * n + l > self.Ne_max):
                # print("Skipping", n, l, twoj)
                # print("Value", el)
                continue
            
            if include_m:
                for twom in product(*[range(-twoj[d], twoj[d] + 1, 2) for d in range(dim)]):
                    for i in range(dim):
                        idx[i] = self.index_flattener(n[i], l[i], twoj[i], twom[i])

                    non_zero = True
                    if m_diagonal:
                        m1 = twom[0:dim // 2]
                        m2 = twom[dim // 2:]
                        if asym:
                            if m1 != m2 and m1 != m2[::-1]:
                                # print(twom, m1, m2)
                                non_zero = False
                        else:
                            if m1 != m2:
                                non_zero = False

                    if not non_zero:
                        flat_matrix[tuple(idx)] = 0
                        continue

                    flat_matrix[tuple(idx)] = el
            else:
                for i in range(dim):
                    idx[i] = self.index_flattener(n[i], l[i], twoj[i], None)
                
                flat_matrix[tuple(idx)] = el
        
        return flat_matrix



if __name__ == '__main__':
    system = System(Ne_max=4, l_max=0, hbar_omega=240, mass=939)

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