from typing import Tuple
import torch
import torchvision
import torchvision.datasets as datasets


def download(root: str) -> Tuple[torchvision.datasets.MNIST]:
    return datasets.MNIST(root=root,
                          train=True,
                          download=True,
                          transform=torchvision.transforms.ToTensor()), \
            datasets.MNIST(root=root,
                           train=False,
                           download=True,
                           transform=torchvision.transforms.ToTensor())


def load_data(batch_size: int, root: str) -> Tuple[torch.utils.data.DataLoader]:
    train_set, test_set = download(root)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=True
    )
    return train_loader, test_loader
