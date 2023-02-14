import numpy as np

# Quantum state class

# Single-particle quantum state
class quantum_state:
    def __init__(self, level, spin, isospin, coefficient=1.0):
        '''
        level: energy level in the shell model 
        spin: spin in the z direction
        isospin: isospin (1/2 is neutron, -1/2 is proton)
        '''
        self.level = level
        self.spin = spin
        self.isospin = isospin
        self.coefficient = coefficient