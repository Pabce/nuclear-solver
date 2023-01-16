import numpy as np
import matplotlib.pyplot as plt

# Define a three-level system (assume orthnormal basis)
E0 = 0.0
E1 = 1.0
E2 = 2.0

class quantum_state:
    def __init__(self, level):
        self.level = level

# Define the potential between two particles
def v(l1, l2):
    return (l1-l2)**2

# Define the action of the potential on a state
def v_state(state1, state2):
    # We will assume non-local potentials because we can
    coef = np.sum(0)
    return v(state1.level, state2.level)

# Compute the matrix elements of the potential
def v_mat():
    v_mat = np.zeros((3 * 3, 3 * 3))
    for l3 in range(3):
        for l4 in range(3):
            for l3 in range(3):
                for l4 in range(3):
                    pass