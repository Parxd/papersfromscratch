import torch
import torchvision
from typing import Tuple
from ..data.download import download

EPOCHS = 10
BATCH_SIZE = 32
IMG_WIDTH = 28
IMG_HEIGHT = 28
NUM_CLASSES = 10
DEST = "./mnist/data"


class MLP(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer1 = torch.nn.Linear(IMG_WIDTH * IMG_HEIGHT, 25)
        self.layer2 = torch.nn.Linear(25, NUM_CLASSES)

    def forward(self, X):
        X = self.layer1(X)
        X = torch.nn.functional.relu(X)
        X = self.layer2(X)
        return torch.nn.functional.softmax(X)


def load_data(root: str) -> Tuple[torch.utils.data.DataLoader]:
    sets: Tuple[torchvision.datasets.MNIST] = download(root)
    train_loader = torch.utils.data.DataLoader(
        dataset=sets[0],
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=sets[1],
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    return train_loader, test_loader


def train():
    load_data(DEST)
    model = MLP()
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        ...
