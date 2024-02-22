import torch
import torchvision
import torchvision.datasets as datasets


def mnist_loader(root='./mnist/data',
                 train=True,
                 batch_size=32,
                 transforms=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])):
    """
    Download MNIST dataset & return a DataLoader.

    Parameters:
    - root (str): Root directory to store the dataset.
    - train (bool): If True, download the training set; else, download the test set.
    - transform (callable, optional): A function/transform

    Returns:
    - DataLoader: PyTorch DataLoader for the MNIST dataset.
    """
    dataset = datasets.MNIST(root=root, train=train, download=True, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
