import random

import numpy as np
import torch


def setup_seed(random_state: int) -> None:
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    np.random.seed(random_state)
    random.seed(random_state)
