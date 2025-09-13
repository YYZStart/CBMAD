import random
import numpy as np
import torch

def setseed(Random_seed):
    random.seed(Random_seed)
    np.random.seed(Random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Random_seed)

    torch.manual_seed(Random_seed)
    torch.cuda.manual_seed(Random_seed)
    torch.cuda.manual_seed_all(Random_seed)


