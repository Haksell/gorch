# ruff: noqa: SIM115

# TODO: batch training

import pickle
from random import gauss

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn

PLOTS = True


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
        self.progress_real = []
        self.progress_generated = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # TODO: clean after batch training
        progress = (
            self.progress_generated if targets.item() == 0 else self.progress_real
        )
        progress.append(loss.item())


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


def plots(discriminator, generator, image_list):
    df = pd.DataFrame(
        np.array(
            [
                discriminator.progress_real,
                discriminator.progress_generated,
                generator.progress,
            ]
        ).T,
        columns=[
            "discriminator loss: real",
            "discriminator loss: generated",
            "generator loss",
        ],
    )
    df.plot(figsize=(16, 8), alpha=0.5, marker=".", grid=True)

    plt.figure(figsize=(16, 8))
    plt.imshow(np.array(image_list).T, interpolation="none", cmap="Reds")

    plt.show()


def main():
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Running on {device_name.upper()}.")

    PKL_GAN = "pkl/gan.pkl"

    try:
        pickled = pickle.load(open(PKL_GAN, "rb"))
        discriminator = pickled["discriminator"].to(device)
        generator = pickled["generator"].to(device)
        image_list = pickled["image_list"]
    except FileNotFoundError:
        discriminator = Discriminator().to(device)
        generator = Generator().to(device)

        target_real = torch.tensor([1.0]).to(device)
        target_generated = torch.tensor([0.0]).to(device)

        # TODO: give different input values to the generator
        generator_input = torch.tensor([0.5]).to(device)

        image_list = []

        EPOCHS = 2000
        for epoch in range(1, EPOCHS + 1):
            discriminator.train(generate_real().to(device), target_real)
            generated = generator.forward(generator_input).detach()
            discriminator.train(generated, target_generated)
            generator.train(discriminator, generator_input, target_real)
            if epoch % 100 == 0 or epoch == EPOCHS:
                print(f"Epoch {epoch}/{EPOCHS}")
                image_list.append(generated.cpu().numpy())

        pickle.dump(
            {
                "discriminator": discriminator,
                "generator": generator,
                "image_list": image_list,
            },
            open(PKL_GAN, "wb"),
        )

    if PLOTS:
        plots(discriminator, generator, image_list)


if __name__ == "__main__":
    main()
