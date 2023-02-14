import numpy as np
import unittest
import matplotlib.pyplot as plt

# Add the src directory to the path
import sys
sys.path.append('../src')

import moshinsky_way as mw


# Test the reduded matrix elements

class TestMatrixElements(unittest.TestCase):
    
    @classmethod
    def setUpClass(self) -> None:
        self.integration_limit = 40
        self.integration_steps = 5500
        self.wavefunctions = mw.set_wavefunctions(Ne_max=10, l_max=4, hbar_omega=3, mass=939,
                                            integration_limit=self.integration_limit, integration_steps=self.integration_steps)

    def test_reduced_matrix_elements(self):
        V0 = 200
        mu = 1.487
        wavefunctions = self.wavefunctions

        # Check for a few cases...
        qn_list = [(0,0,0,0), (0,1,0,1), (1,0,1,0), (1,1,0,1), (1,1,1,0), (1,2,0,2), (2,5,3,1), (2,2,2,2), (3,3,3,3), (4,4,4,4),
                    (3,2,1,0), (1,2,3,4), (1,1,1,1)]
        
        math_results = [1.99865, 0.0927275, 2.73055, 0.139813, 0.69469, 0.00767507, 0.00108347, 0.0280917, 0.00404346, 0.000605482, 0.279794,
                        0.00184658, 0.211007]

        for i, quantum_numbers in enumerate(qn_list):
            n1, l1, n2, l2 = quantum_numbers
            el = mw.central_potential_reduced_matrix_element(wavefunctions, V0, mu, n1, l1, n2, l2, self.integration_limit, self.integration_steps)

            self.assertTrue(np.isclose(el, math_results[i], atol=1e-5))


if __name__ == '__main__':
    # t = TestMatrixElements()
    # t.test_reduced_matrix_elements()
    unittest.main()
