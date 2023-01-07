import numpy as np


def generate_rand_array(max: int, sample: int, random_state: int) -> np.array:
    np.random.seed(random_state)
    return np.random.choice(np.arange(0, max), sample, replace=False)
