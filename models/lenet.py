"""
LeNet-5 as described in: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

Input1: Input layer
    - H: 32
    - W: 32
    - F: 1

Conv2: 2D convolution
    - Input filters: 1 (greyscale inputs)
    - Output filters: 6
    - Spatial extent: 5
    - Stride: 1
    - Padding: 0

    - Output shape
        generally (where W_new is width of feature map)
        - W_new = (W_old - F + 2P) / S + 1
        - H_new = (H_old - F + 2P) / S + 1
        - D_new = K

        - W: ((32 - 5 + 2(0)) / 1) + 1 = 28
        - H = ((32 - 5 * 2(0)) / 1) + 1 = 28
        - D = 6

Subsample3: 2D subsample

Conv4: 2D convolution

Subsample5: 2D subsample

"""

import torch


class LeNet5(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c1 = torch.nn.Conv2d(in_channels=1,
                                  out_channels=6,
                                  kernel_size=5,
                                  stride=1)
        self.s2 = torch.nn.AvgPool2d(kernel_size=2)
        self.c3 = ...

    def forward(self, X):
        X = self.c1(X)
        X = self.s2(X)
        # X = self.c3(X)
        return X
