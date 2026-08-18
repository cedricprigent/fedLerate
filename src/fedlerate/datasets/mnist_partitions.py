import numpy as np
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from fedlerate.data.transforms import Rotate, get_normalization
from fedlerate.data.partition import generate_partitions

def generate_mnist_partitions(alpha, n_clients):
    
    # Load train & test sets
    trainset = MNIST(root="/tmp/app/data", train=True, download=True)
    testset = MNIST(root="/tmp/app/data", train=False, download=False)

    clients_per_cluster = n_clients // 4

    # Prepare transforms for each client
    transform = [transforms.Compose([
        transforms.ToPILImage(),
        Rotate((client // clients_per_cluster) * 90),
        transforms.ToTensor(),
        get_normalization(channels=1)
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