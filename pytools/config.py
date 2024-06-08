# System imports
import random

# Third-party imports
import numpy as np
import torch

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False