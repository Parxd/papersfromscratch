import torch


class MLP(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer1 = torch.nn.Linear(28 * 28, 512)
        self.layer2 = torch.nn.Linear(512, 256)
        self.layer3 = torch.nn.Linear(256, 10)

    def forward(self, X):
        X = X.reshape(32, 1, 784)
        X = self.layer1(X)
        X = torch.nn.functional.leaky_relu(X)
        X = self.layer2(X)
        X = torch.nn.functional.leaky_relu(X)
        return self.layer3(X)
