'''
Standalone implementation of the shell model to solve for a given nucleus' ground state, using the harmonic oscillator basis.
'''

import numpy as np
import matplotlib.pyplot as plt
from findiff import FinDiff
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from scipy.linalg import eigh
import qutip as qt

import harmonic_plotter as hp


# CONSTANTS (in Natural units)
HBAR = 1 # (action)
C = 1 # (speed of light)
EPS_0 = 1 # (permittivity of free space)

ESQ_OVER_FOURPI = 1/137.036 # (electric constant) (adim)
E = np.sqrt(4 * np.pi / 137.036) # (electron charge) (adim)
N_MASS = 939.565 # MeV
P_MASS = 938.272 # MeV
AMU = 931.494 # MeV
FM = 1/197 # MeV^-1
# ---------

# Number of nucleons
A = 33
Z = 16
N = A - Z

# We use the Seminole parametrization of the Woods-Saxon potential
# Reduced masses
MU_P = (1/P_MASS + 1/((A-1)*AMU)) ** -1 
MU_N = (1/N_MASS + 1/((A-1)*AMU)) ** -1
# Woods-Saxon potential parameters
r0 = 1.26 * FM # fm -> MeV^-1
r0_so = 1.16 * FM # fm -> MeV^-1
a = 0.662 * FM # fm -> MeV^-1
V0 = 52.06  # MeV 
kappa = 0.639 # dimensionless (isospin splitting)
# Spin-orbit coupling parameters
lamb = 24.1 # dimensionless
# ---------------------------------


# Woods-Saxon potential
def woods_saxon(r, A, Z, r0, a, V0, kappa, particle_type):
    R0 = r0 * A**(1 / 3)

    denom = 1 + ((r - R0) * 1/a).expm()
    V = -V0 * denom.inv()
    
    # Isospin splitting factor
    V = V * isospin_factor(A, Z, kappa, particle_type)

    return V

# Spin-orbit coupling potential
def spin_orbit(r, j, l, A, r0_so, a, V0, lamb, particle_type):
    # If l = 0, there is no spin-orbit coupling. We introduce a "hard wall" at r ~ 0 (you know why!)
    if l == 0:
        wall = np.zeros(r.shape)
        wall[0] = 1e15
        return wall

    R0 = r0_so * A**(1 / 3)

    # Derivative of the Woods-Saxon potential (with a sign change)
    dV = +V0 * ((r - R0) * 1/a).expm() * (a * (1 + ((r - R0) * 1/a).expm()) **2).inv()

    # l * s term
    ls = 0.5 * (j * (j + 1) - l * (l + 1) - 3/4)

    # Reduced mass
    rm = MU_P if particle_type == 'proton' else MU_N

    return -lamb * (2 * rm**2 * r).inv() * (HBAR/C)**2 * dV * ls

# Coulomb potential
def coulomb(r, r0, A, Z, particle_type):
    coulomb_potential = np.zeros(r.shape)
    if particle_type == 'neutron':
        return coulomb_potential

    R0 = r0 * A**(1 / 3)

    coulomb_potential[r < R0] = ESQ_OVER_FOURPI * (Z - 1) / (2 * R0) * (3 - (r[r < R0] / R0)**2)
    coulomb_potential[r >= R0] = ESQ_OVER_FOURPI * (Z - 1) / r[r >= R0]

    return coulomb_potential

# Isospin factor (adimensional)
def isospin_factor(A, Z, kappa, particle_type):
    N = A - Z
    prefactor = kappa/A
    mult = 1 if particle_type == 'proton' else -1

    if N == Z:
        return 1 + 3 * prefactor
    elif N > Z:
        return 1 + (mult * (N - Z + 1) + 2) * prefactor
    elif N < Z:
        return 1 + (mult * (N - Z - 1) + 2) * prefactor

# Total potential for an individual nucleon
def potential(r, j, l, A, Z, particle_type, r0, a, V0, r0_so, lamb, kappa):
    
    ws = woods_saxon(r, A, Z, r0, a, V0, kappa, particle_type)
    so = spin_orbit(r, j, l, A, r0_so, a, V0, lamb, particle_type)
    #cou = coulomb(r, r0, A, Z, particle_type)
    
    return ws #+ so #+ cou

# We need to define an effective potential for the radial part of the wave function
def effective_potential(r, j, l, A, Z, particle_type, r0, a, V0, r0_so, lamb, kappa):
    V = potential(r, j, l, A, Z, particle_type, r0, a, V0, r0_so, lamb, kappa)
    rm = MU_P if particle_type == 'proton' else MU_N

    V_eff = l * (l + 1) * (2 * rm * r**2).inv() + V

    return V_eff


# We are going to try and solve in a harmonic oscillator basis. Wish me luck...
def harmonic_basis():
    n = 1
    l = 0
    j = l + 1/2
    nucleon = "neutron"

    # Harmonic oscillator basis size (has to be even so r is invertible)
    basis_size = 300
    # Reduced mass
    red_mass = MU_P if nucleon == 'proton' else MU_N
    omega = np.sqrt(2 * V0 / ((r0 * A**(1 / 3))**2 * red_mass)) # The results should be independent from this

    # Laplacian operator (using qtip) (we use creation and destruction because I'm not sure of the normalization)
    phat = (0 + 1j) * np.sqrt(red_mass * omega / 2) * (qt.create(basis_size) - qt.destroy(basis_size))
    lap = - phat * phat
    # Position operator
    rhat = 1/np.sqrt(2 * red_mass * omega) * (qt.destroy(basis_size) + qt.create(basis_size))
    # Potential operator in terms of the position operator
    vhat = effective_potential(rhat, j, l, A, Z, nucleon, r0, a, V0, r0_so, lamb, kappa)

    # Hamiltonian
    H = - lap / (2 * red_mass) + vhat

    #print(H)
    #print(vhat.check_herm())

    eigenvalues = H.eigenenergies().real
    eigenstates = H.eigenstates()
    print(eigenvalues[0:12])
    #print(np.diff(eigenvalues)[0:12])

    # print(H.eigenstates()[0])

    # wf = 0
    # x = np.arange(-10, 10, 0.1)
    # for i, coef in enumerate(eigenstates[0]):
    #     wf += coef * hp.stationary_state(x, i, m=red_mass, omega=omega)

    # plt.plot(x, wf)
    # plt.show()


if __name__ == '__main__':

    harmonic_basis()
