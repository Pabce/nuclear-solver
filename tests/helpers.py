import numpy as np


def random_quantum_number_generator(Ne_max, l_level_max, sample_size, set_size, seed=None):
    """
    Generates a list of random quantum numbers for the given Ne_max and l_level_max
    """

    # Use this for consistent testing
    if seed:
        np.random.seed(seed)
    
    # Generate random n and Ne. This will serve to limit the l values
    qn_list_n = np.random.randint(0, Ne_max // 2 + 1, size=(sample_size, set_size))
    max_Ne_list = np.minimum(Ne_max, 2 * qn_list_n + l_level_max)
    qn_list_Ne = np.random.randint(2* qn_list_n, max_Ne_list + 1, size=(sample_size, set_size))

    # Generate random l values
    qn_list_l = qn_list_Ne - 2 * qn_list_n

    # Generate random j values
    qn_list_twoj_idx = np.random.randint(0, 2, size=(sample_size, set_size))
    qn_list_twoj = 2 * qn_list_l - 1 + qn_list_twoj_idx * 2
    # Filter the j=-1/2 unphysical values
    qn_list_twoj = np.maximum(qn_list_twoj, 1)

    # Concatenate the quantum numbers
    qn_list = np.concatenate((qn_list_n, qn_list_l, qn_list_twoj), axis=1)

    # print(np.all(max_Ne_list <= Ne_max))
    # print(np.all(qn_list_l <= l_level_max))
    # print(np.all(qn_list_twoj >= 1))

    return qn_list, qn_list_twoj_idx


if __name__ == "__main__":
    random_quantum_number_generator(6, 4, 100, 4)