'''
Solve the Schrôdinger equation in the 3D harmonic oscillator basis
'''

import numpy as np
import matplotlib.pyplot as plt
import small_s_solver as ss
from scipy.special import genlaguerre, factorial2
from scipy.integrate import quad
from scipy.linalg import eigh
from numba import cfunc
from numba.types import intc, CPointer, float64, int32, int64
from numpy.polynomial.hermite import Hermite
from scipy.special import genlaguerre, factorial2, binom

#from shell import effective_potential as shellpot
from shell import potential as shellpot
from shell import WS_PARAMETERS, FM, S, P_MASS, N_MASS, AMU, V0

# hbar = 1
# omega = 1

# Define the potential
def effective_potential(r, l, omega, mass):
    return 0.5 * mass * omega**2 * r**2 + l*(l+1)/(2*mass*r**2)


# Retrun the energies
def energy(k=-1, n=-1, l=0, omega=1):
    if k > 0:
        return omega * (2*k + l + 1.5)
    elif n > 0:
        return omega * (n + 1.5)

    return -1

# Return the radial wavefunctions
# This must take the input IN FERMIS
def wavefunction(r, k=-1, n=-1, l=0, omega=1, mass=1):
    if k < 0:
        k = int((n - l)/2)

    a_kl = np.sqrt(2**(k + l + 2) * np.math.factorial(k) / (factorial2(2*k + 2*l + 1) * np.pi**0.5))
    b = np.sqrt(1/(mass * omega))
    squiggle = r / b
    laguerre = genlaguerre(n=k, alpha=l+0.5)

    r_kl = a_kl * b**(-1.5) * np.exp(-squiggle**2 / 2) * squiggle**l * laguerre(squiggle**2)

    # norm = np.sum(np.abs(r_kl)**2 * r**2) * (r[1] - r[0])
    # print(norm)

    return r_kl

# Wavefunction expressed as a power series, avoids all sorts of weird functions and is accurate a.f.
def series_wavefunction(r, k, l, omega, mass):
    b = np.sqrt(1/(mass * omega))
    squiggle = r / b

    wf = np.zeros(r.shape)
    a_0 = 1
    a_prev = a_0

    for i in range(0, 30):
        if i == 0:
            a = a_0
        else:
            a = a_prev * ((i-1) - k) / (((i-1) + 1) * (l + (i-1) + 3/2))

        wf += a * squiggle ** (l + 2 * i)

        a_prev = a

    wf = wf * np.exp(-squiggle**2 / 2)
    
    # Normalize the wavefunction
    norm = np.sum(np.abs(wf)**2 * r**2) * (r[1] - r[0])
    wf /= np.sqrt(norm)
    
    return wf, np.sqrt(norm)

# As this will be called for an indiviual value of r, we have to precompute the normalization
@cfunc(float64(float64, float64, float64, float64, float64, float64))
def series_wavefunction_val(r, k, l, omega, mass, sqrt_norm):
    b = np.sqrt(1/(mass * omega))
    squiggle = r / b

    wf = 0
    a_0 = 1
    a_prev = a_0

    for i in range(0, 30):
        if i == 0:
            a = a_0
        else:
            a = a_prev * ((i-1) - k) / (((i-1) + 1) * (l + (i-1) + 3/2))

        wf += a * squiggle ** (l + 2 * i)

        a_prev = a
    
    wf = wf * np.exp(-squiggle**2 / 2)

    # Normalize the wavefunction
    wf /= sqrt_norm
    
    return wf



# Calculate matrix elements (for operators dependent on r only)
def matrix_element(r, k1, l1, m1, k2, l2, m2, operator, omega=1, mass=1, print=False):
    # psi1_c = lambda r: np.conj(wavefunction(r, k=k1, l=l1, omega=omega, mass=mass))
    # psi2 = lambda r: wavefunction(r, k=k2, l=l2, omega=omega, mass=mass)

    # # Function to integrate (inline)
    # func = lambda r, psi1_c, psi2, operator: psi1_c(r) * operator(r) * psi2(r)
    # result, error = quad(func, 0.00001, 20, args=(psi1_c, psi2, operator))
    # print(result, error)

    # Due to spherical harmonics orthogonality:
    if m1 != m2 or l1 != l2:
        return 0
    l = l1
    m = m1

    psi1_c = np.conj(wavefunction(r, k=k1, l=l, omega=omega, mass=mass))
    psi2 = wavefunction(r, k=k2, l=l, omega=omega, mass=mass)
    
    result = np.sum(psi1_c * operator * psi2 * np.diff(r, append=0))

    if print:
        plt.figure(1)
        plt.plot(r / FM, psi1_c)
        plt.plot(r / FM, psi2)

        plt.figure(2)
        plt.plot(r / FM, operator)
        plt.ylim(-60, 50)

        plt.figure(3)
        plt.plot(r / FM, psi1_c * operator * psi2)
        plt.show()

    return result


# Pretty sure this is wrong!
def kinetic_energy_matrix_element(r, k1, l1, m1, k2, l2, m2, particle_type, omega=1, mass=1):
    # Due to spherical harmonics orthogonality:
    if m1 != m2 or l1 != l2:
        return 0
    l = l1
    m = m1

    psi1_c = np.conj(wavefunction(r, k=k1, l=l, omega=omega, mass=mass))
    psi2 = wavefunction(r, k=k2, l=l, omega=omega, mass=mass)

    f = psi1_c
    h = r[1] - r[0] 
    n = f.shape[0]
    lap = np.zeros(n)
    lap[0] = (f[1] - 2*f[0] + f[n-1]) / h**2
    lap[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / h**2
    lap[-1] = (f[0] - 2*f[-1] + f[-2]) / h**2

    red_mass = MU_P if particle_type == 'proton' else MU_N
    result = np.sum(-lap / (2 * red_mass) * psi2 * np.diff(r, append=0))

    return result


def build(r, omega, mass, particle_type, k_num=2, l_num=2):
    # Number of "k" levels
    k_num = k_num
    l_num = l_num 
    # Ignore different m levels for now, as degenerate

    # Build the potential matrix and kinetic matrix
    pot_matrix = np.zeros((k_num * l_num * 2, k_num * l_num * 2))
    kin_matrix = np.zeros((k_num * l_num * 2, k_num * l_num * 2))

    # Matrix elements for l1 != l2 will vanish
    for k1 in range(k_num):
        for l in range(l_num):
            for n_j in range(0, 2):
                j = l + (n_j - 1) * 0.5 # j = l +/- 1/2
                operator = shellpot(r, l, j, 33, 16, particle_type, *WS_PARAMETERS)
                for k2 in range(k_num):
                    elx = k1 * l_num * 2 + l + n_j
                    ely = k2 * l_num * 2 + l + n_j
                    
                    pot_matrix[elx, ely] = matrix_element(r, k1, l, 0, k2, l, 0, operator, omega=omega, mass=mass)

                    if k1 == k2:
                        kin = 0.5 * omega * (2 * k1 + l + 1.5)
                    elif k1 == k2 - 1:
                        kin = 0.5 * omega * np.sqrt(k2 * (k2 + l + 0.5))
                    elif k1 == k2 + 1:
                        kin = 0.5 * omega * np.sqrt(k1 * (k1 + l + 0.5))
                    else:
                        kin = 0

                    kin_matrix[elx, ely] = kin

    print(kin_matrix)
    # Hamiltonian matrix
    ham_matrix = pot_matrix + kin_matrix

    return ham_matrix
    

if __name__ == "__main__":
    #solver = ss.S_solver(effective_potential)

    # Number of nucleons
    A = 33
    Z = 16
    N = A - Z
    
    R0 = 1.26 * (A ** (1/3)) * FM

    # We use the Seminole parametrization of the Woods-Saxon potential
    # Reduced masses
    MU_P = (1/P_MASS + 1/((A-1)*AMU)) ** -1 
    MU_N = (1/N_MASS + 1/((A-1)*AMU)) ** -1
    particle_type = 'neutron'

    # Discretized space
    r = np.linspace(0.001 * FM, 15 * FM, 2500)

    # Parameters
    mass = MU_P if particle_type == 'proton' else MU_N
    omega = np.sqrt(2 * V0 / (R0**2 * mass))

    #n = 2
    k1 = 0
    k2 = 0
    l = 1
    m = 0

    operator = shellpot(r, l, l+1/2, 33, 16, particle_type, *WS_PARAMETERS)

    el = matrix_element(r, k1, l, m, k2, l, m, operator, omega=omega, mass=mass, print=False)
    print(el)

    hamiltonian = build(r, omega, mass, particle_type, k_num=2, l_num=2)
    print(hamiltonian)

    # Solve the eigenvalue problem
    eigenvalues, eigenvectors = eigh(hamiltonian, subset_by_index=[0,7])

    print(eigenvalues)


    # Check mormalization 
    print(np.sum(np.abs(wavefunction(r, k=k1, l=l, omega=omega, mass=mass) * r)**2 * np.diff(r, append=0)))
    plt.plot(r / FM, np.abs(wavefunction(r, k=k1, l=l, omega=omega, mass=mass))**2)
    plt.show()

    


    

