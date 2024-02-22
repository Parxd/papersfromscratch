import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from ..data.download import mnist_loader


LR = 0.01
MOMENTUM = 0.9
EPOCHS = 5
BATCH_SIZE = 32
IMG_WIDTH = 28
IMG_HEIGHT = 28
NUM_CLASSES = 10
ROOT = "./mnist/data"


class MLP(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer1 = torch.nn.Linear(IMG_WIDTH * IMG_HEIGHT, 512)
        self.layer2 = torch.nn.Linear(512, 256)
        self.layer3 = torch.nn.Linear(256, NUM_CLASSES)

    def forward(self, X):
        X = X.reshape(32, 1, 784)
        X = self.layer1(X)
        X = torch.nn.functional.leaky_relu(X)
        X = self.layer2(X)
        X = torch.nn.functional.leaky_relu(X)
        return self.layer3(X)


def train():
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5,), (0.5,))
    ])
    train_loader = mnist_loader(root=ROOT, train=True, transforms=transforms)
    model = MLP()
    model.to('cuda')
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    criterion = torch.nn.CrossEntropyLoss()
    
    # y_hat = model(next(iter(train_loader))[0])  # 32 x 10
    # y = torch.tensor(
    #     [0, 8, 1, 1, 3, 0, 2, 2, 3, 1, 1, 4, 6, 0, 6, 3, 9, 1, 5, 8, 1, 3, 2, 5, 0, 5, 9, 2, 5, 8, 8, 3]
    #     )  # 1 x 32, values ranging from (0, NUM_Y_HAT_COLS - 1)
    # print(criterion(y_hat.squeeze(axis=1), y))
    # print(y_hat.shape)

    for epoch in range(EPOCHS):
        for img, label in train_loader:
            img, label = img.to('cuda'), label.to('cuda')
            # CE loss expecting class indices, not probabilities
            out = model(img).squeeze(axis=1)
            loss = criterion(out, label)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch} loss={loss.item()}")
 