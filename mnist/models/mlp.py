import torch
from ..data.download import load_data

EPOCHS = 10
BATCH_SIZE = 32
IMG_WIDTH = 28
IMG_HEIGHT = 28
NUM_CLASSES = 10
ROOT = "./mnist/data"


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


def train():
    train_loader, test_loader = load_data(BATCH_SIZE, ROOT)
    model = MLP()
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        ...
    item = next(iter(train_loader))
