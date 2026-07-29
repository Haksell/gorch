from random import gauss

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn

PLOTS = False


def generate_real():
    STD = 0.1
    return torch.tensor([gauss(1, STD), gauss(0, STD), gauss(1, STD), gauss(0, STD)])


def generate_random():
    return torch.rand(4)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 3),
            nn.Sigmoid(),
            nn.Linear(3, 1),
            nn.Sigmoid(),
        )
        self.loss_function = nn.MSELoss()  # TODO: BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters())
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.progress.append(loss.item())


def plots(discriminator):
    df = pd.DataFrame(discriminator.progress, columns=["loss"])
    df.plot(ylim=(0, 1), figsize=(16, 8), alpha=0.5, marker=".", grid=True)
    plt.show()


def main():
    discriminator = Discriminator()
    for _ in range(10000):
        discriminator.train(generate_real(), torch.tensor([1.0]))
        discriminator.train(generate_random(), torch.tensor([0.0]))

    with torch.no_grad():
        print(discriminator.forward(generate_real()).item())
        print(discriminator.forward(generate_random()).item())

    if PLOTS:
        plots(discriminator)


if __name__ == "__main__":
    main()
