# ruff: noqa: SIM115

import pickle

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

EPOCHS = 8
BATCH_SIZE = 256

HIDDEN_NODES = 128

PLOTS = False
TEST_IDX = 33


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, HIDDEN_NODES),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(HIDDEN_NODES),
            nn.Linear(HIDDEN_NODES, 10),
        )
        self.loss_function = nn.CrossEntropyLoss()
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


class MNIST(Dataset):
    def __init__(self, csv_file, device):
        # TODO: don't use pandas (and probably not numpy either)
        df = pd.read_csv(csv_file, header=None)

        images = df.iloc[:, 1:].to_numpy(dtype="float32", copy=True) / 255
        self.images = torch.from_numpy(images).to(device)

        labels = df.iloc[:, 0].to_numpy(dtype="int64", copy=True)
        self.labels = torch.from_numpy(labels).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def plot_image(self, index):
        img = self.images[index].reshape(28, 28).to("cpu")
        plt.title(f"label = {self.labels[index]}")
        plt.imshow(img, interpolation="none", cmap="Blues")


def test_accuracy(classifier, mnist_test):
    score = items = 0
    for image, label in mnist_test:
        with torch.no_grad():
            output = classifier.forward(image.unsqueeze(0))
            answer = output.detach().cpu().numpy().argmax()
        if answer == label.item():
            score += 1
        items += 1
    print(f"Accuracy: {score}/{items} = {score / items:.2%}")


def plots(classifier, mnist_test):
    image, _ = mnist_test[TEST_IDX]

    with torch.no_grad():
        output = classifier.forward(image.unsqueeze(0))
        output = F.softmax(output, dim=1).squeeze()

    pd.DataFrame(output.detach().cpu().numpy()).plot(
        kind="bar", legend=False, ylim=(0, 1)
    )
    plt.show()

    mnist_test.plot_image(TEST_IDX)
    plt.show()

    df = pd.DataFrame(classifier.progress, columns=["loss"])
    df.plot(ylim=(0, 1), figsize=(16, 8), alpha=0.5, marker=".", grid=True)
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {str(device).upper()}.")

    PKL_TRAIN = "pkl/mnist_train.pkl"
    PKL_TEST = "pkl/mnist_test.pkl"
    PKL_CLASSIFIER = "pkl/classifier.pkl"

    try:
        mnist_train = pickle.load(open(PKL_TRAIN, "rb"))
    except FileNotFoundError:
        mnist_train = MNIST("data/mnist_train.csv", device)
        pickle.dump(mnist_train, open(PKL_TRAIN, "wb"))

    try:
        mnist_test = pickle.load(open(PKL_TEST, "rb"))
    except FileNotFoundError:
        mnist_test = MNIST("data/mnist_test.csv", device)
        pickle.dump(mnist_test, open(PKL_TEST, "wb"))

    try:
        classifier = pickle.load(open(PKL_CLASSIFIER, "rb")).to(device)
        test_accuracy(classifier, mnist_test)
    except FileNotFoundError:
        classifier = Classifier().to(device)
        data_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
        for epoch in range(1, EPOCHS + 1):
            print(f"Training epoch {epoch}/{EPOCHS}...")
            for images, labels in data_loader:
                classifier.train(images, labels)
            test_accuracy(classifier, mnist_test)
        pickle.dump(classifier, open(PKL_CLASSIFIER, "wb"))

    if PLOTS:
        plots(classifier, mnist_test)


if __name__ == "__main__":
    main()
