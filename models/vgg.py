"""
VGG-16D and VGG-19E as described in: https://arxiv.org/pdf/1409.1556v6.pdf
"""
import torch


class VGG16(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c1 = torch.nn.Conv2d(in_channels=3,
                                  out_channels=64,
                                  kernel_size=3)
        
        self.c2 = torch.nn.Conv2d(in_channels=64,
                                  out_channels=64,
                                  kernel_size=3)
        
        self.m3 = torch.nn.MaxPool2d(kernel_size=2,
                                     stride=2)
        
        self.c4 = torch.nn.Conv2d(in_channels=64,
                                  out_channels=128,
                                  kernel_size=64)
        
        self.c5 = torch.nn.Conv2d(in_channels=128,
                                  out_channels=128,
                                  kernel_size=3)
        
        self.m6 = torch.nn.MaxPool2d(kernel_size=2,
                                     stride=2)
        
        self.c6 = torch.nn.Conv2d(in_channels=128,
                                  out_channels=256,
                                  kernel_size=3)
        
        self.c7 = torch.nn.Conv2d(in_channels=256,
                                  out_channels=256,
                                  kernel_size=3)
        
        self.c8 = torch.nn.Conv2d(in_channels=256,
                                  out_channels=256,
                                  kernel_size=3)
        
        self.m9 = torch.nn.MaxPool2d(kernel_size=2,
                                     stride=2)
        
        self.c10 = torch.nn.Conv2d(in_channels=256,
                                   out_channels=512,
                                   kernel_size=3)
        
        self.c11 = torch.nn.Conv2d(in_channels=512,
                                   out_channels=512,
                                   kernel_size=3)
        
        self.c12 = torch.nn.Conv2d(in_channels=512,
                                   out_channels=512,
                                   kernel_size=3)

        self.m13 = torch.nn.MaxPool2d(kernel_size=2,
                                      stride=2)
        
        self.c14 = torch.nn.Conv2d(in_channels=512,
                                   out_channels=512,
                                   kernel_size=3)
        
        self.c15 = torch.nn.Conv2d(in_channels=512,
                                   out_channels=512,
                                   kernel_size=3)
        
        self.c16 = torch.nn.Conv2d(in_channels=512,
                                   out_channels=512,
                                   kernel_size=3)
        
        self.m17 = torch.nn.MaxPool2d(kernel_size=2,
                                      stride=2)
        
        self.f18 = torch.nn.Linear(4096, 4096)
        self.f19 = torch.nn.Linear(4096, 4096)
        self.f20 = torch.nn.Linear(4096, 1000)
        
    def forward(self, X):
        ...
