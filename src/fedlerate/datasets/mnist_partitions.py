import numpy as np
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from fedlerate.data.transforms import Rotate, MinMaxScaler
from fedlerate.data.partition import generate_partitions

def generate_mnist_partitions():
    
    n_clients = 100
    alpha = 1
    
    # Load train & test sets
    trainset = MNIST(root="/tmp/app/data", train=True, download=True)
    testset = MNIST(root="/tmp/app/data", train=False, download=False)

    # Prepare transforms for each client
    transform = [transforms.Compose([
        Rotate((client // 25) * 90),
        transforms.Lambda(lambda image: torch.tensor(np.array(image).astype(np.float32))),
        MinMaxScaler(min_value=0, max_value=255)
    ]) for client in range(n_clients)]

    # Generate client partitions
    generate_partitions(
        trainset=trainset,
        testset=testset,
        n_partitions=n_clients,
        alpha=alpha,
        transform=transform,
        seed=0
    )