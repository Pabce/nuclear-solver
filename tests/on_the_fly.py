'''
On-the-fly tests of the generated matrices during the generation process and the Hartree-Fock calculation.
'''

import numpy as np
from termcolor import colored, cprint


def test_run_matrices(system, rho=None, D=None, hamiltonian=None, gamma=None, include_m=False):
    '''
    Test the matrices generated during the Hartree-Fock calculation.
    '''

    if rho is not None:
        size = rho.shape[0]
        if include_m:
            rho_comp_squared = np.allclose(rho, np.dot(rho, rho))
            cprint("RHO equal to RHO^2: {}".format(rho_comp_squared), "green" if rho_comp_squared else "red")
        else:
            # TODO: implement a test for rho in the case of no m
            pass

        rho_hermitian = np.allclose(rho, rho.conj().T)
        cprint("RHO hermitian: {}".format(rho_hermitian), "green" if rho_hermitian else "red")
    
    if D is not None:
        size = D.shape[0]
        D_unitary = np.allclose(np.dot(D, D.conj().T), np.eye(size))
        cprint("D unitary: {}".format(D_unitary), "green" if D_unitary else "red")
    
    if gamma is not None:
        size = gamma.shape[0]
        gamma_hermitian = np.allclose(gamma, gamma.conj().T)
        cprint("Gamma hermitian: {}".format(gamma_hermitian), "green" if gamma_hermitian else "red")

    if hamiltonian is not None:
        size = hamiltonian.shape[0]
        hamiltonian_hermitian = np.allclose(hamiltonian, hamiltonian.conj().T)
        cprint("Hamiltonian hermitian: {}".format(hamiltonian_hermitian), "green" if hamiltonian_hermitian else "red")
        if not hamiltonian_hermitian:
            #print(hamiltonian - hamiltonian.conj().T)
            print(np.linalg.norm(hamiltonian - hamiltonian.conj().T))
    

    # Test the diagonalities in l, j, (m)
    rho_diag_l, rho_diag_lj, rho_diag_ljm = True, True, True
    D_diag_l, D_diag_lj, D_diag_ljm = True, True, True
    hamiltonian_diag_l, hamiltonian_diag_lj, hamiltonian_diag_ljm = True, True, True
    for k1 in range(size):
        for k2 in range(size):
            if include_m:
                n1, l1, twoj1, twom1 = system.index_unflattener(k1)
                n2, l2, twoj2, twom2 = system.index_unflattener(k2)
            else:
                n1, l1, twoj1 = system.index_unflattener(k1)
                n2, l2, twoj2 = system.index_unflattener(k2)


            if l1 != l2:
                if rho is not None:
                    if not np.allclose(rho[k1, k2], 0):
                        #print(rho[k1, k2])
                        # print("RHO NOT DIAGONAL IN L! k1, k2:", k1, k2)
                        # print(system.index_unflattener(k1), system.index_unflattener(k2))
                        rho_diag_l = False
                if D is not None:
                    if not np.allclose(D[k1, k2], 0):
                        # print("D NOT DIAGONAL IN L! k1, k2:", k1, k2)
                        #print(D[k1, k2], l1, l2)
                        # print(system.index_unflattener(k1), system.index_unflattener(k2))
                        D_diag_l = False
                if hamiltonian is not None:
                    if not np.allclose(hamiltonian[k1, k2], 0):
                        # print("HAMILTONIAN NOT DIAGONAL IN L! k1, k2:", k1, k2)
                        # print(system.index_unflattener(k1), system.index_unflattener(k2))
                        hamiltonian_diag_l = False
            
            elif twoj1 != twoj2:
                if rho is not None:
                    if not np.allclose(rho[k1, k2], 0):
                        rho_diag_lj = False
                if D is not None:
                    if not np.allclose(D[k1, k2], 0):
                        D_diag_lj = False
                if hamiltonian is not None:
                    if not np.allclose(hamiltonian[k1, k2], 0):
                        hamiltonian_diag_lj = False
            
            elif include_m and twom1 != twom2:
                if rho is not None:
                    if not np.allclose(rho[k1, k2], 0):
                        rho_diag_ljm = False
                if D is not None:
                    if not np.allclose(D[k1, k2], 0):
                        D_diag_ljm = False
                if hamiltonian is not None:
                    if not np.allclose(hamiltonian[k1, k2], 0):
                        hamiltonian_diag_ljm = False
            
            # if twoj1 != twoj2:
            #     print(rho[k1, k2] == 0)

    if rho is not None:
        cprint("RHO diagonal in l: {}".format(rho_diag_l), "green" if rho_diag_l else "red")
        cprint("RHO diagonal in l, j: {}".format(rho_diag_lj), "green" if rho_diag_lj else "red")
        if include_m:
            cprint("RHO diagonal in l, j, m: {}".format(rho_diag_ljm), "green" if rho_diag_ljm else "red")

    if D is not None:
        cprint("D diagonal in l: {}".format(D_diag_l), "green" if D_diag_l else "yellow")
        cprint("D diagonal in l, j: {}".format(D_diag_lj), "green" if D_diag_lj else "yellow")
        if include_m:
            cprint("D diagonal in l, j, m: {}".format(D_diag_ljm), "green" if D_diag_ljm else "red")

    if hamiltonian is not None:
        cprint("Hamiltonian diagonal in l: {}".format(hamiltonian_diag_l), "green" if hamiltonian_diag_l else "red")
        cprint("Hamiltonian diagonal in l, j: {}".format(hamiltonian_diag_lj), "green" if hamiltonian_diag_lj else "red")
        if include_m:
            cprint("Hamiltonian diagonal in l, j, m: {}".format(hamiltonian_diag_ljm), "green" if hamiltonian_diag_ljm else "red")


    # Check that D respects l, j symmetry rowwise


def test_matrix_elements(V_matrizx=None, t_matrix=None):
    pass