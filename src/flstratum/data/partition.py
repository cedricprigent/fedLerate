from typing import Dict, List, Tuple
from PIL import Image
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader

from flstratum.data.transforms import MinMaxScaler


def dirichlet_partitions(
    dataset: Dataset,
    num_clients: int,
    alpha: float,
    transform=None
) -> List[Dataset]:
    
    NUM_CLASS = len(dataset.classes)
    MIN_SIZE = 0
    X = [[] for _ in range(num_clients)]
    Y = [[] for _ in range(num_clients)]
    if not isinstance(dataset.targets, np.ndarray):
        dataset.targets = np.array(dataset.targets, dtype=np.int64)
    idx = [np.where(dataset.targets == i)[0] for i in range(NUM_CLASS)]

    # Generate client partitions from dirichlet distributions
    while MIN_SIZE < 10:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(NUM_CLASS):
            np.random.shuffle(idx[k])
            distributions = np.random.dirichlet(np.repeat(alpha, num_clients))
            distributions = np.array(
                [
                    p * (len(idx_j) < len(dataset) / num_clients)
                    for p, idx_j in zip(distributions, idx_batch)
                ]
            )
            distributions = distributions / distributions.sum()
            distributions = (np.cumsum(distributions) * len(idx[k])).astype(int)[:-1]
            idx_batch = [
                np.concatenate((idx_j, idx.tolist())).astype(np.int64)
                for idx_j, idx in zip(idx_batch, np.split(idx[k], distributions))
            ]

            for i in range(num_clients):
                np.random.shuffle(idx_batch[k])
                X[i] = dataset.data[idx_batch[i]]
                Y[i] = dataset.targets[idx_batch[i]]

        MIN_SIZE = min([len(y) for y in Y])

    if transform is not None:
        datasets = [
            torch.utils.data.TensorDataset(
                transform[j](X[j]),
                torch.from_numpy(Y[j])
            ) for j in range(num_clients)
        ]
    else:
        datasets = [
            torch.utils.data.TensorDataset(
                X[j],
                torch.from_numpy(Y[j])
            ) for j in range(num_clients)
        ]

    counts = []
    for data in datasets:
        missing_classes = 0
        for label in range(10):
            idx = torch.where(data.tensors[1] == label)
            if (len(idx[0]) == 0):
                missing_classes += 1
        counts.append(missing_classes)
    for i in range(10):
        print(f"{counts.count(i)} clients missing {i} classes")
            

    return datasets



def generate_partitions(trainset, testset, n_partitions, alpha, transform, seed=None):
    os.makedirs('/tmp/app/data/train', exist_ok=True)
    os.makedirs('/tmp/app/data/test', exist_ok=True)

    if seed is not None:
        np.random.seed(seed)

    train_subsets = dirichlet_partitions(dataset=trainset, num_clients=n_partitions, alpha=alpha, transform=transform)
    test_subsets = dirichlet_partitions(dataset=testset, num_clients=n_partitions, alpha=alpha, transform=transform)

    # Save subsets
    for i, (train_subset, test_subset) in enumerate(zip(train_subsets, test_subsets)):
        print(f"subset {i}: {len(train_subset)} train data, {len(test_subset)} test data")

        torch.save(train_subset, f"/tmp/app/data/train/train_subset-{i}.pth")
        torch.save(test_subset, f"/tmp/app/data/test/test_subset-{i}.pth")



def load_partition(num_partition, batch_size=16, transform=None, target_transform=None):
    """Load generated train and test subset"""
    """Generating partitions with partition_data.py is needed beforehand"""
    # Load train & test subsets
    trainset = torch.load(f"/tmp/app/data/train/train_subset-{num_partition}.pth", weights_only=False)
    testset = torch.load(f"/tmp/app/data/test/test_subset-{num_partition}.pth", weights_only=False)

    trainset.target_transform = target_transform
    testset.target_transform = target_transform
    trainset.transform = transform
    testset.transform = transform

    # Generate dataloaders
    trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True
        )

    testloader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=True
        )

    num_examples = {"trainset": len(trainloader), "testset": len(testloader)}
    return trainloader, testloader, num_examples


class FolderPartition(Dataset):
    def __init__(self, data, targets, transform = None):
        self.loader = default_loader
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        sample = self.loader(data[index])
        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target