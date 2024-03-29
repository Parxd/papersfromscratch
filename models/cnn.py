import torch


class CNN_small(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2)
        self.layer2 = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=1)
        
        self.layer3 = torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3)
        self.layer4 = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=1)
        
        self.layer5 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=8)
    
        self.layer6 = torch.nn.Linear(196, 50)
        self.layer7 = torch.nn.Linear(50, 10)

    def forward(self, X):
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        X = self.layer5(X)
        X = X.reshape(32, 1, 196)
        X = self.layer6(X)
        X = torch.nn.functional.leaky_relu(X)
        X = self.layer7(X)
        return X
