'''
Standalone implementation of the shell model to solve for a given nucleus' ground state.
'''

import numpy as np
import matplotlib.pyplot as plt
from findiff import FinDiff
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from scipy.linalg import eigh
import qutip as qt


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
S = 6.58e22 # MeV^-1
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

WS_PARAMETERS = [r0, a, V0, r0_so, lamb, kappa]
# ---------------------------------

# Harmonic potential
def ws_harmonic(r, r0, A, particle_type):
    R0 = r0 * A**(1 / 3)

    red_mass = MU_P if particle_type == 'proton' else MU_N
    omega2 = 2 * V0 / (R0**2 * red_mass)
    
    return 0.5 * red_mass * omega2 * (r**2 - R0**2)


# Woods-Saxon potential
def woods_saxon(r, A, Z, r0, a, V0, kappa, particle_type):
    R0 = r0 * A**(1 / 3)

    V = -V0 / (1.0 + np.exp((r - R0) / a))
    
    # Isospin splitting factor
    V *= isospin_factor(A, Z, kappa, particle_type)

    return V

# Spin-orbit coupling potential
def spin_orbit(r, l, j, A, r0_so, a, V0, lamb, particle_type):
    # If l = 0, there is no spin-orbit coupling. We introduce a "hard wall" at r ~ 0 (you know why!)
    if l == 0:
        wall = np.zeros(r.shape)
        wall[0] = 1e15
        return wall

    R0 = r0_so * A**(1 / 3)

    # Derivative of the Woods-Saxon potential (with a sign change)
    dV = +V0 * np.exp((r - R0) / a) / (a * (1 + np.exp((r - R0) / a))**2)

    # l * s term
    ls = 0.5 * (j * (j + 1) - l * (l + 1) - 3/4)

    # Reduced mass
    rm = MU_P if particle_type == 'proton' else MU_N

    return -lamb * 1/(2 * rm**2 * r) * (HBAR/C)**2 * dV * ls

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
def potential(r, l, j, A, Z, particle_type, r0, a, V0, r0_so, lamb, kappa):
    
    ws = woods_saxon(r, A, Z, r0, a, V0, kappa, particle_type)
    so = spin_orbit(r, l, j, A, r0_so, a, V0, lamb, particle_type)
    cou = coulomb(r, r0, A, Z, particle_type)
    
    return ws + so + cou


# We need to define an effective potential for the radial part of the wave function
def effective_potential(r, l, j, A, Z, particle_type, r0, a, V0, r0_so, lamb, kappa):
    V = potential(r, l, j, A, Z, particle_type, r0, a, V0, r0_so, lamb, kappa)
    rm = MU_P if particle_type == 'proton' else MU_N

    V_eff = HBAR**2 * l * (l + 1) / (2 * rm * r**2) + V

    return V_eff



if __name__ == '__main__':
    # Quantum numbers
    #n = 0
    l = 1
    j = l + 1/2
    nucleon = "neutron"

    # Radial grid
    r_min = 0.0001 * FM
    r_max = 15 * FM
    n_steps = 1200
    r = np.linspace(r_min, r_max, n_steps)
    
    # plt.plot(r / FM, ws_harmonic(r, r0, A, nucleon), label="Harmonic", linestyle="dashdot")
    # plt.plot(r / FM, woods_saxon(r, A, Z, r0, a, V0, kappa, nucleon), label="Woods-Saxon", linestyle="dashed")
    # plt.plot(r / FM, spin_orbit(r, j, l, A, r0_so, a, V0, lamb, nucleon), label="Spin-orbit", linestyle="dashed")
    # plt.plot(r / FM, coulomb(r, r0, A, Z, nucleon), label="Coulomb", linestyle="dashed")

    # plt.plot(r / FM, potential(r, j, l, A, Z, nucleon, r0, a, V0, r0_so, lamb, kappa), label="Total")
    # plt.plot(r / FM, effective_potential(r, j, l, A, Z, nucleon, r0, a, V0, r0_so, lamb, kappa), label="Effective")
    
    # plt.ylim(-V0 - 10, 50)
    # plt.legend()
    # plt.show()

    # Solve the radial Schrödinger equation
    # Create the Hamiltonian
    # Laplacian operator:
    lap = FinDiff(0, r, 2).matrix(r.shape)
    red_mass = MU_P if nucleon == 'proton' else MU_N

    hamiltonian = -lap / (2 * red_mass) + np.diag(effective_potential(r, l, j, A, Z, nucleon, r0, a, V0, r0_so, lamb, kappa))
    eigenvalues, eigenvectors = eigh(hamiltonian, subset_by_index=[0,10])

    # hamiltonian = -lap / (2 * red_mass) + diags(effective_potential(r, j, l, A, Z, nucleon, r0, a, V0, r0_so, lamb, kappa))
    # eigenvalues, eigenvectors = eigs(hamiltonian, k=10, which='SR') # Seems slower!

    # Remove eigenvalues smaller than V0 (they are spurious)
    eigenvectors = eigenvectors[:, eigenvalues > -V0]
    eigenvalues = eigenvalues[eigenvalues > -V0]

    #print(eigenvectors[:3, 0])
    print(eigenvalues)

    # Plot the wave function (you have to multiply by r to get back the proper wave function!)
    norm = np.sqrt(np.sum(np.abs(eigenvectors[:, 0])**2 * np.diff(r)[0]))
    eigenvectors[:, 0] = eigenvectors[:, 0] / norm

    print(np.sum(np.abs(eigenvectors[:, 0])**2 * np.diff(r)[0]))
    plt.plot(r / FM, eigenvectors[:, 0] / r, label="Wave function")
    plt.show()


