"""
LeNet-5 as described in: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
"""

import torch


INPUT_H = 32
INPUT_W = 32


class LeNet5(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c1 = torch.nn.Conv2d(in_channels=1,
                                  out_channels=6,
                                  kernel_size=5,
                                  stride=1)
        self.s2 = torch.nn.AvgPool2d(kernel_size=2)

    def forward(self, X):
        X = self.c1(X)
        X = self.s2(X)
        return X

    def layers(self):
        return 