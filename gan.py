# TODO: batch training

from random import gauss

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn

PLOTS = False


def generate_real():
    STD = 0.1
    return torch.tensor([gauss(1, STD), gauss(0, STD), gauss(1, STD), gauss(0, STD)])


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


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 3),
            nn.Sigmoid(),
            nn.Linear(3, 4),
            nn.Sigmoid(),
        )
        self.optimizer = torch.optim.Adam(self.parameters())
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, discriminator, inputs, targets):
        g_output = self.forward(inputs)
        d_output = discriminator.forward(g_output)
        loss = discriminator.loss_function(d_output, targets)
        self.progress.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def plots(discriminator):
    df = pd.DataFrame(discriminator.progress, columns=["loss"])
    df.plot(ylim=(0, 1), figsize=(16, 8), alpha=0.5, marker=".", grid=True)
    plt.show()


def main():
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Running on {device_name.upper()}.")

    discriminator = Discriminator().to(device)
    generator = Generator().to(device)

    target_real = torch.tensor([1.0]).to(device)
    target_generated = torch.tensor([0.0]).to(device)

    # TODO: give different input values to the generator
    generator_input = torch.tensor([0.5]).to(device)

    EPOCHS = 10_000
    for epoch in range(1, EPOCHS + 1):
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}")
        discriminator.train(generate_real().to(device), target_real)
        # TODO: merge these two steps
        discriminator.train(generator.forward(generator_input), target_generated)
        generator.train(discriminator, generator_input, target_real)

    if PLOTS:
        plots(discriminator)


if __name__ == "__main__":
    main()
