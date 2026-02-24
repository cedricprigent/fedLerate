import argparse
import json

import numpy as np
import torch
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms

from fedlerate.data.transforms import Rotate, MinMaxScaler
from fedlerate.data.partition import generate_partitions

if __name__ == "__main__":
    # --- Arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='FL/configs/mnist-alpha-1.json')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)

    dataset = config['data']['dataset']
    n_clients = config['data']['n_clients']
    n_classes = config['data']['n_classes']
    alpha = config['data']['alpha']
    output_dim = config['model']['output_dim']
    n_clusters = config['clustering']['n_clusters']
    distance_metric = config['clustering']['distance_metric']
    n_iter = config['clustering']['n_iterations']

    # Load train & test sets
    if dataset == "mnist":
        dataset = MNIST
    elif dataset == "cifar10":
        dataset = CIFAR10
    else:
        raise NotImplementedError

    trainset = dataset(root="/tmp/app/data", train=True, download=True)
    testset = dataset(root="/tmp/app/data", train=False, download=False)

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