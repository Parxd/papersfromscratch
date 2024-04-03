"""
LeNet-5 as described in: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

General output feature map shape calculations (from https://cs231n.github.io/convolutional-networks/)
- Input feature map: (W_old, H_old, D_old)
- Output map: (W_new, H_new, D_new)
- F: spatial extent
- S: stride
- P: padding

    - 2D convolution
        - W_new = (W_old - F + 2P) / S + 1
        - H_new = (H_old - F + 2P) / S + 1
        - D_new = K

    - 2D sub-sample
        - W_new = (W_old - F) / S + 1
        - H_new = (H_old - F) / S + 1
        - D_new = D_old

LeNet-5 layers
--------------
I1: Input layer
    - W: 32
    - H: 32
    - D: 1

C2: 2D convolution
    - Input filters: 1 (greyscale inputs)
    - Output filters: 6
    - Spatial extent: 5
    - Stride: 1
    - Padding: 0

    - Output shape
        - W = ((32 - 5 + 2(0)) / 1) + 1 = 28
        - H = ((32 - 5 * 2(0)) / 1) + 1 = 28
        - D = 6

S3: 2D sub-sample
    - Input filters: 6
    - Output filters: 6
    - Spatial extent: 2
    - Stride: 2
    - Padding: 0
    
    - Output shape
        - W = ((28 - 2) / 2) + 1 = 14
        - H = ((28 - 2) / 2) + 1 = 14
        - D = 6

C4: 2D convolution
    - Input filters: 6
    - Output filters: 16
    - Spatial extent: 5
    - Stride: 1
    - Padding: 0

S5: 2D sub-sample

"""

import torch
from typing import Tuple


# custom pooling layer, basically an avg. pool but with parameters
class Subsample(torch.nn.Module):
    def __init__(self, kernel_size: int | Tuple, stride: int, padding: int = 0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.w = torch.rand(1, requires_grad=True)
        self.b = torch.rand(1, requires_grad=True)
        self.spatial_extent = kernel_size if isinstance(kernel_size, Tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

    def forward(self, X):
        avg = torch.nn.functional.avg_pool2d(X, self.spatial_extent, self.stride, self.padding)
        return self.w * (avg * self.spatial_extent[0] * self.spatial_extent[1]) + self.b


class LeNet5(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.c1 = torch.nn.Conv2d(in_channels=1,
                                  out_channels=6,
                                  kernel_size=5,
                                  stride=1)

        self.s2 = Subsample(kernel_size=2, stride=2)
        
        self.c3 = torch.nn.Conv2d(in_channels=6,
                                  out_channels=16,
                                  kernel_size=5,
                                  stride=1)
        
        self.s4 = Subsample(kernel_size=2, stride=2)
        
        self.c5 = torch.nn.Linear()
        
    def forward(self, X):
        X = self.c1(X)
        X = self.s2(X)
        X = self.c3(X)
        X = self.s4(X)
        X = self.c5(X)
        return X
