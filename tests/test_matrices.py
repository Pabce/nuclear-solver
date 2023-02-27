import numpy as np
import unittest
import matplotlib.pyplot as plt
from itertools import product

# Add the src directory to the path
import sys
sys.path.append('../src')

import moshinsky_way as mw # type: ignore
import helpers


class TestMatrices(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        self.Ne_max, self.l_max = 4, 4
        self.integration_limit = 40
        self.integration_steps = 5500
        self.wavefunctions = mw.set_wavefunctions(Ne_max=self.Ne_max, l_max=self.l_max, hbar_omega=3, mass=939,
                                            integration_limit=self.integration_limit, integration_steps=self.integration_steps)


        self.moshinsky_brackets = mw.set_moshinsky_brackets(self.Ne_max, self.l_max)
        self.wigner_9j_dict = mw.set_wigner_9js()

        self.central_potential_reduced_matrix = mw.set_central_potential_reduced_matrix(self.wavefunctions, 200, 1.487,
                                                 self.Ne_max, self.l_max, self.integration_limit, self.integration_steps)
        self.central_potential_ls_coupling_basis_matrix = mw.set_central_potential_ls_coupling_basis_matrix(
                                                    self.central_potential_reduced_matrix, self.moshinsky_brackets, self.Ne_max, self.l_max)
        
        self.central_potential_J_coupling_matrix = mw.set_central_potential_J_coupling_basis_matrix(
                                                        self.central_potential_ls_coupling_basis_matrix,self.wigner_9j_dict,
                                                        self.Ne_max, self.l_max, spin_selector="singlet", parity_selector="even")
        
        self.central_potential_matrix = mw.set_central_potential_matrix(self.central_potential_J_coupling_matrix, self.Ne_max, self.l_max)
    

    def test_reduced_matrix(self):
        # The reduced matrix should be diagonal in l
        for n1, l1, n2, l2 in product(range(self.Ne_max//2+1), range(self.l_max+1), range(self.Ne_max//2+1), range(self.l_max+1)):
            if l1 != l2:
                self.assertTrue(np.isclose(self.central_potential_reduced_matrix[n1,l1,n2,l2], 0))

        # Check if the reduced matrix is symmetric in the exchange of (n1, l1)<->(n2, l2)
        for n1, n2, l in product(range(self.Ne_max//2+1), range(self.Ne_max//2+1), range(self.l_max+1)):
            self.assertTrue(np.isclose(self.central_potential_reduced_matrix[n1,l,n2,l],
                                        self.central_potential_reduced_matrix[n2,l,n1,l]))
        
        # Check if the matrix values coincide with the values returned by the matrix element function
        for n1, n2, l in product(range(self.Ne_max//2+1), range(self.Ne_max//2+1), range(self.l_max+1)):
            val = mw.central_potential_reduced_matrix_element(self.wavefunctions, n1, l, n2, l, 200, 1.487, self.integration_limit, self.integration_steps)
            mat_val = self.central_potential_reduced_matrix[n1,l,n2,l]

            if not np.isclose(val, mat_val):
                print(val, mat_val)
                print(n1, n2, l)

            self.assertTrue(np.isclose(val, mat_val))

    

    def test_ls_coupling_basis_matrix(self):
        # Generate some random n values
        sample_size = 50
        qn_list, _ = helpers.random_quantum_number_generator(self.Ne_max, self.l_max, sample_size, 4, seed=123)

        # for i in range(sample_size):
        #     n1, n2, n3, n4 = qn_list[i, 0:4]

        #     for l1, l2, l3, l4 in product(range(self.l_max+1), repeat=4):
        #         if (l1, l2) != (l3, l4):
        #             lamb_max = min(l1 + l2, l3 + l4)
        #             lamb_min = max(abs(l1 - l2), abs(l3 - l4))
        #             for lamb in range(lamb_min, lamb_max + 1):
        #                 self.assertTrue(np.isclose(self.central_potential_ls_coupling_basis_matrix[n1, l1, n2, l2, n3, l3, n4, l4, lamb], 0))

        # ls coupling basis matrix should be symmetric in the exchange of (n1, l1, n2, l2) <-> (n3, l3, n4, l4)
        for i in range(sample_size):
            n1, n2, n3, n4 = qn_list[i, 0:4]
            l1, l2, l3, l4 = qn_list[i, 4:8]
            
            lamb_max = min(l1 + l2, l3 + l4)
            lamb_min = max(abs(l1 - l2), abs(l3 - l4))
            for lamb in range(lamb_min, lamb_max + 1):
                self.assertTrue(np.isclose(self.central_potential_ls_coupling_basis_matrix[n1, l1, n2, l2, n3, l3, n4, l4, lamb],
                                            self.central_potential_ls_coupling_basis_matrix[n3, l3, n4, l4, n1, l1, n2, l2, lamb]))

        
    def test_J_coupling_basis_matrix(self):
        # Generate some random n values
        sample_size = 50
        qn_list, qn_two_idx = helpers.random_quantum_number_generator(self.Ne_max, self.l_max, sample_size, 4, seed=123)

        # J coupling basis matrix should be symmetric in the exchange of (n1, l1, j1, n2, l2, j2) <-> (n3, l3, j3, n4, l4, j4)
        for i in range(sample_size):
            n1, n2, n3, n4 = qn_list[i, 0:4]
            l1, l2, _, _ = qn_list[i, 4:8]
            twoj1, twoj2, _, _ = qn_list[i, 8:12]
            twoj1_idx, twoj2_idx, _, _= qn_two_idx[i, :]
            l3, l4 = l1, l2
            twoj3, twoj4 = twoj1, twoj2
            twoj3_idx, twoj4_idx = twoj1_idx, twoj2_idx
            
            for twoJ in range(np.abs(twoj1 - twoj2), twoj1 + twoj2 + 1, 2):
                self.assertTrue(np.isclose(self.central_potential_J_coupling_matrix[n1, l1, twoj1_idx, n2, l2, twoj2_idx, n3, l3, twoj3_idx, n4, l4, twoj4_idx, twoJ],
                                            self.central_potential_J_coupling_matrix[n3, l3, twoj3_idx, n4, l4, twoj4_idx, n1, l1, twoj1_idx, n2, l2, twoj2_idx, twoJ]))

    def test_central_potential_matrix(self):
        # Generate some random n values
        sample_size = 50
        qn_list, qn_two_idx = helpers.random_quantum_number_generator(self.Ne_max, self.l_max, sample_size, 4, seed=123)

        # CP matrix should be symmetric in the exchange of (n1, l1, j1, n2, l2, j2) <-> (n3, l3, j3, n4, l4, j4)
        for i in range(sample_size):
            n1, n2, n3, n4 = qn_list[i, 0:4]
            l1, l2, _, _ = qn_list[i, 4:8]
            twoj1, twoj2, _, _ = qn_list[i, 8:12]
            twoj1_idx, twoj2_idx, _, _= qn_two_idx[i, :]
            l3, l4 = l1, l2
            twoj3, twoj4 = twoj1, twoj2
            twoj3_idx, twoj4_idx = twoj1_idx, twoj2_idx
     
            self.assertTrue(np.isclose(self.central_potential_matrix[n1, l1, twoj1_idx, n2, l2, twoj2_idx, n3, l3, twoj3_idx, n4, l4, twoj4_idx],
                                            self.central_potential_matrix[n3, l3, twoj3_idx, n4, l4, twoj4_idx, n1, l1, twoj1_idx, n2, l2, twoj2_idx]))
