import numpy as np
import unittest
import matplotlib.pyplot as plt
from itertools import product

# Add the src directory to the path
import sys
sys.path.append('../src')

import moshinsky_way as mw # type: ignore
import helpers


class TestMatrixElements(unittest.TestCase):
    
    @classmethod
    def setUpClass(self) -> None:
        self.Ne_max, self.l_max = 4, 4
        self.integration_limit = 40
        self.integration_steps = 5500
        self.wavefunctions = mw.set_wavefunctions(Ne_max=self.Ne_max, l_max=self.l_max, hbar_omega=3, mass=939,
                                            integration_limit=self.integration_limit, integration_steps=self.integration_steps)


        self.moshinsky_brackets = mw.set_moshinsky_brackets(self.Ne_max, self.l_max)
        self.central_potential_reduced_matrix = mw.set_central_potential_reduced_matrix(self.wavefunctions, 200, 1.487,
                                                 self.Ne_max, self.l_max, self.integration_limit, self.integration_steps)
        self.central_potential_ls_coupling_basis_matrix = mw.set_central_potential_ls_coupling_basis_matrix(
                                                    self.central_potential_reduced_matrix, self.moshinsky_brackets, self.Ne_max, self.l_max)
        
        self.wigner_9j_dict = mw.set_wigner_9js()


    # Test the reduced matrix elements
    # TODO: this is potential specific, so generalize it
    def test_reduced_matrix_elements_value(self):
        V0 = 200
        mu = 1.487
        wavefunctions = self.wavefunctions

        # Check for a few cases...
        qn_list = [(0,0,0,0), (0,1,0,1), (1,0,1,0), (1,1,0,1), (1,1,1,0), (1,2,0,2), (2,5,3,1), (2,2,2,2), (3,3,3,3), (4,4,4,4),
                    (3,2,1,0), (1,2,3,4), (1,1,1,1)]
        
        math_results = [1.99865, 0.0927275, 2.73055, 0.139813, 0.69469, 0.00767507, 0.00108347, 0.0280917, 0.00404346, 
                        0.000605482, 0.279794, 0.00184658, 0.211007]

        for i, quantum_numbers in enumerate(qn_list):
            n1, l1, n2, l2 = quantum_numbers
            el = mw.central_potential_reduced_matrix_element(wavefunctions, V0, mu, n1, l1, n2, l2, 
                                                                self.integration_limit, self.integration_steps)

            self.assertTrue(np.isclose(el, math_results[i], atol=1e-5))
    
    def test_reduced_matrix_elements_symmetry(self):
        # The reduced matrix elements should be symmetric in the exchange of (n1, l1)<->(n2, l2)
        V0 = 200
        mu = 1.487
        qn_list = [(1,1,0,1), (1,1,1,0), (1,2,0,2), (2,5,3,1), (3,2,1,0), (1,2,3,4), (2,1,1,2), (2,0,2,1), (1,0,0,2)]

        for quantum_numbers in qn_list:
            n1, l1, n2, l2 = quantum_numbers
            el1 = mw.central_potential_reduced_matrix_element(self.wavefunctions, V0, mu, n1, l1, n2, l2, 
                                                                    self.integration_limit, self.integration_steps)
            el2 = mw.central_potential_reduced_matrix_element(self.wavefunctions, V0, mu, n2, l2, n1, l1, 
                                                                    self.integration_limit, self.integration_steps)

            self.assertTrue(np.isclose(el1, el2, atol=1e-12))
    
    # Test the ls coupling basis matrix elements
    def test_ls_coupling_matrix_elements_symmetry(self):
        # The ls coupling matrix elements should be symmetric in the exchange of (n1, l1)<->(n2, l2)

        # Generate random quantum numbers
        sample_size = 30
        qn_list, _ = helpers.random_quantum_number_generator(self.Ne_max, self.l_max, sample_size, set_size=4, seed=12)

        for qns in range(sample_size):
            n1, n2, n3, n4, l1, l2, l3, l4, _, _, _, _ = qn_list[qns, :]

            lamb_max = min(l1 + l2, l3 + l4)
            lamb_min = max(abs(l1 - l2), abs(l3 - l4))

            for lamb in range(lamb_min, lamb_max + 1):
                el1 = mw.central_potential_ls_coupling_basis_matrix_element(self.central_potential_reduced_matrix, self.moshinsky_brackets,
                                                                                n1, l1, n2, l2, n3, l3, n4, l4, lamb)
                el2 = mw.central_potential_ls_coupling_basis_matrix_element(self.central_potential_reduced_matrix, self.moshinsky_brackets,
                                                                                n3, l3, n4, l4, n1, l1, n2, l2, lamb)

                self.assertTrue(np.isclose(el1, el2, atol=1e-12))
    
    # Test the J-basis matrix elements
    def test_J_basis_matrix_elements_symmetry(self):
        # The J coupling matrix elements should be symmetric in the exchange of (n1, l1, j1)<->(n2, l2, j2)

        # Generate random quantum numbers
        sample_size = 30
        qn_list, _ = helpers.random_quantum_number_generator(self.Ne_max, self.l_max, sample_size, set_size=4, seed=12)

        for qns in range(sample_size):
            n1, n2, n3, n4, l1, l2, _, _, twoj1, twoj2, _, _ = qn_list[qns, :]
            # Lazy approach (remember the structure of the matrix):
            l3, l4 = l1, l2
            twoj3, twoj4 = twoj1, twoj2

            selectors = product(["none", "singlet", "triplet"], ["none", "even", "odd"])

            for sel in selectors:
                spin_selector, parity_selector = sel
                for twoJ in range(abs(twoj1 - twoj2), twoj1 + twoj2 + 1, 2):
                    el1 = mw.central_potential_J_coupling_matrix_element(self.central_potential_ls_coupling_basis_matrix,
                                                    self.wigner_9j_dict, n1, l1, twoj1, n2, l2, twoj2,
                                                    n3, l3, twoj3, n4, l4, twoj4, twoJ, spin_selector, parity_selector)

                    el2 = mw.central_potential_J_coupling_matrix_element(self.central_potential_ls_coupling_basis_matrix,
                                                    self.wigner_9j_dict, n3, l3, twoj3, n4, l4, twoj4,
                                                    n1, l1, twoj1, n2, l2, twoj2, twoJ, spin_selector, parity_selector)

                    #print(type(el1), type(el2))

                    self.assertTrue(np.isclose(el1, el2, atol=1e-12))
    

    def test_J_basis_matrix_elements_spin_selectors(self):
        # Spin selected matrix elements (singlet and triplet) should add up to the total matrix element
        sample_size = 30
        qn_list, _ = helpers.random_quantum_number_generator(self.Ne_max, self.l_max, sample_size, set_size=4, seed=12)

        for qns in range(sample_size):
            n1, n2, n3, n4, l1, l2, _, _, twoj1, twoj2, _, _ = qn_list[qns, :]
            # Lazy approach (remember the structure of the matrix):
            l3, l4 = l1, l2
            twoj3, twoj4 = twoj1, twoj2

            for twoJ in range(abs(twoj1 - twoj2), twoj1 + twoj2 + 1, 2):
                el1 = mw.central_potential_J_coupling_matrix_element(self.central_potential_ls_coupling_basis_matrix,
                                                self.wigner_9j_dict, n1, l1, twoj1, n2, l2, twoj2,
                                                n3, l3, twoj3, n4, l4, twoj4, twoJ, "none", "none")

                el2 = mw.central_potential_J_coupling_matrix_element(self.central_potential_ls_coupling_basis_matrix,
                                                self.wigner_9j_dict, n1, l1, twoj1, n2, l2, twoj2,
                                                n3, l3, twoj3, n4, l4, twoj4, twoJ, "singlet", "none")

                el3 = mw.central_potential_J_coupling_matrix_element(self.central_potential_ls_coupling_basis_matrix,
                                                self.wigner_9j_dict, n1, l1, twoj1, n2, l2, twoj2,
                                                n3, l3, twoj3, n4, l4, twoj4, twoJ, "triplet", "none")

                self.assertTrue(np.isclose(el1, el2 + el3, atol=1e-12))



if __name__ == '__main__':
    # t = TestMatrixElements()
    # t.test_J_basis_matrix_elements_symmetry()
    unittest.main()

    pass
