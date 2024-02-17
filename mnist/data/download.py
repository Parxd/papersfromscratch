from typing import Tuple
import torchvision
import torchvision.datasets as datasets


def download(root: str) -> Tuple[torchvision.datasets.MNIST]:
    return datasets.MNIST(root=root, train=True, download=True, transform=None), \
            datasets.MNIST(root=root, train=False, download=True, transform=None)
