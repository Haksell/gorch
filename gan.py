from random import gauss

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn


def generate_real():
    STD = 0.1
    return torch.tensor([gauss(1, STD), gauss(0, STD), gauss(1, STD), gauss(0, STD)])


print(generate_real())
