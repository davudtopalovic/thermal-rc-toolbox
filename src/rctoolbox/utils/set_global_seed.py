import numpy as np
import random

def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)