# TODO: batch training with a DataLoader

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

DEVICE = "cuda"


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.Sigmoid(),  # TODO: ReLU
            nn.Linear(200, 10),
            nn.Sigmoid(),  # TODO: Softmax?
        )
        self.loss_function = nn.MSELoss()  # TODO: CrossEntropyLoss
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)  # TODO: Adam
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
        if self.counter % 1000 == 0:
            print(f"{self.counter=}")

    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=["loss"])
        df.plot(
            ylim=(0, 1),
            figsize=(16, 8),
            alpha=0.1,
            marker=".",
            grid=True,
            yticks=(0, 0.25, 0.5),
        )


class MNIST(Dataset):
    def __init__(self, csv_file):
        # TODO: don't use pandas (and probably not numpy either)
        df = pd.read_csv(csv_file, header=None)

        images = df.iloc[:, 1:].to_numpy(dtype="float32", copy=True) / 255
        self.images = torch.from_numpy(images).to(DEVICE)

        labels = df.iloc[:, 0].to_numpy(dtype="int64", copy=True)
        self.labels = torch.from_numpy(labels)
        self.targets = F.one_hot(self.labels, 10).to(DEVICE)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.labels[index], self.images[index], self.targets[index]

    def plot_image(self, index):
        img = self.images[index].reshape(28, 28).to("cpu")
        plt.title(f"label = {self.labels[index]}")
        plt.imshow(img, interpolation="none", cmap="Blues")


def main():
    assert torch.cuda.is_available()

    PKL_TRAIN = "pkl/mnist_train.pkl"
    PKL_TEST = "pkl/mnist_test.pkl"
    PKL_CLASSIFIER = "pkl/classifier.pkl"

    try:
        mnist_train = pickle.load(open(PKL_TRAIN, "rb"))
    except FileNotFoundError:
        mnist_train = MNIST("data/mnist_train.csv")
        pickle.dump(mnist_train, open(PKL_TRAIN, "wb"))

    try:
        mnist_test = pickle.load(open(PKL_TEST, "rb"))
    except FileNotFoundError:
        mnist_test = MNIST("data/mnist_test.csv")
        pickle.dump(mnist_test, open(PKL_TEST, "wb"))

    try:
        classifier = pickle.load(open(PKL_CLASSIFIER, "rb"))
    except FileNotFoundError:
        EPOCHS = 3
        classifier = Classifier().to(DEVICE)
        for epoch in range(1, EPOCHS + 1):
            print(f"Training epoch {epoch}/{EPOCHS}")
            for _, image_data_tensor, target_tensor in mnist_train:
                classifier.train(image_data_tensor, target_tensor)
        pickle.dump(classifier, open(PKL_CLASSIFIER, "wb"))

    TEST_IDX = 33
    image_data = mnist_test[TEST_IDX][1]
    output = classifier.forward(image_data)

    pd.DataFrame(output.detach().cpu().numpy()).plot(
        kind="bar", legend=False, ylim=(0, 1)
    )
    plt.show()

    mnist_test.plot_image(TEST_IDX)
    plt.show()

    classifier.plot_progress()
    plt.show()


if __name__ == "__main__":
    main()
