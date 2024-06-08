# Standard library imports
import os

# Third-party imports
import numpy as np
import torch


if __name__ == "__main__":
    type = "cutfree"
    version = "v1"

    dir = os.path.dirname(os.path.realpath(__file__))
    path = dir + f"/{type}-{version}"

    vocab = torch.load(f"{path}/vocab.pt")
    print(np.array(vocab.itos), "\n")

    classes = torch.load(f"{path}/classes.pt")
    print(classes, "\n")

    hyperparameters = torch.load(f"{path}/hyperparameters.pt")
    for key, value in hyperparameters.items():
        print(f"{key}: {value}")