import matplotlib
import matplotlib.pyplot as plt 
import numpy
import numpy.polynomial.hermite as Herm
import math

#Choose simple units
hbar=1.

def hermite(x, n, m, omega):
    xi = numpy.sqrt(m * omega/hbar)*x
    herm_coeffs = numpy.zeros(n+1)
    herm_coeffs[n] = 1
    return Herm.hermval(xi, herm_coeffs)
  
def stationary_state(x,n, m, omega):
    xi = numpy.sqrt(m*omega/hbar)*x
    prefactor = 1./math.sqrt(2.**n * math.factorial(n)) * (m*omega/(numpy.pi*hbar))**(0.25)
    psi = prefactor * numpy.exp(- xi**2 / 2) * hermite(x,n,m,omega)
    return psi


if __name__ == '__main__':
    #Discretized space
    dx = 0.05
    x_lim = 12
    x = numpy.arange(-x_lim,x_lim,dx)

    plt.figure()
    plt.plot(x, stationary_state(x,4, m=1, omega=1))
    plt.xlabel(r"x")
    plt.ylabel(r"$\psi_4(x)$")
    plt.title(r"Test Plot of $\psi_4(x)$")
    plt.show()