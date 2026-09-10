import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose, ToTensor, Resize
from fedlerate.data.transforms import get_normalization

DOMAINS = ['photo', 'art_painting', 'cartoon', 'sketch']

class CombinedPACSDataset(Dataset):
    def __init__(self, datasets: List[ImageFolder]):
        self.samples = []
        self.targets = []
        self.domain_targets = []
        
        for domain_idx, ds in enumerate(datasets):
            for path, target in ds.samples:
                self.samples.append(path)
                self.targets.append(target)
                self.domain_targets.append(domain_idx)

        self.targets = np.array(self.targets, dtype=np.int64)
        self.domain_targets = np.array(self.domain_targets, dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx], self.domain_targets[idx]


def dirichlet_domain_partitions(
    combined_dataset: CombinedPACSDataset,
    total_clients: int,
    alpha: float,
) -> Tuple[List[List[str]], List[List[int]], List[List[int]]]:
    """Partitions the dataset across clients using a Dirichlet distribution over DOMAIN labels."""
    np.random.seed(0)
    
    NUM_DOMAINS = len(DOMAINS)
    MIN_SIZE = 0
    X = [[] for _ in range(total_clients)]
    Y = [[] for _ in range(total_clients)]
    D = [[] for _ in range(total_clients)]

    idx_by_domain = [np.where(combined_dataset.domain_targets == i)[0] for i in range(NUM_DOMAINS)]

    while MIN_SIZE < 20:
        idx_batch = [[] for _ in range(total_clients)]
        for k in range(NUM_DOMAINS):
            np.random.shuffle(idx_by_domain[k])
            
            distributions = np.random.dirichlet(np.repeat(alpha, total_clients))
            distributions = np.array(
                [
                    p * (len(idx_j) < len(combined_dataset) / total_clients)
                    for p, idx_j in zip(distributions, idx_batch)
                ]
            )
            distributions = distributions / distributions.sum()
            distributions = (np.cumsum(distributions) * len(idx_by_domain[k])).astype(int)[:-1]
            
            idx_batch = [
                np.concatenate((idx_j, idx.tolist())).astype(np.int64)
                for idx_j, idx in zip(idx_batch, np.split(idx_by_domain[k], distributions))
            ]
            MIN_SIZE = min([len(idx_j) for idx_j in idx_batch])

    for i in range(total_clients):
        np.random.shuffle(idx_batch[i])
        X[i] = [combined_dataset.samples[k] for k in idx_batch[i]]
        Y[i] = [combined_dataset.targets[k] for k in idx_batch[i]]
        D[i] = [combined_dataset.domain_targets[k] for k in idx_batch[i]]

    return X, Y, D


def generate_pacs_mixed_partitions(total_clients=40, alpha=0.1, root='/tmp/app/data/pacs_data/'):
    """Generates partitions where each client receives a soft mixture of PACS domains."""
    os.makedirs('/tmp/app/data/train', exist_ok=True)
    os.makedirs('/tmp/app/data/test', exist_ok=True)

    paths = [f'{root}/{d}/' for d in DOMAINS]

    transform = Compose([ToTensor(), Resize(size=(64, 64), antialias=True), get_normalization(channels=3)])
    raw_datasets = [ImageFolder(p) for p in paths]
    combined_dataset = CombinedPACSDataset(raw_datasets)

    print("#################################################")
    print("# Generating MIXED Domain partitions for PACS\t#")
    print(f"# - domain alpha = {alpha}\t\t\t\t#")
    print(f"# - total clients = {total_clients}\t\t\t#")
    print("#################################################\n")

    client_paths, client_targets, client_domains = dirichlet_domain_partitions(
        combined_dataset=combined_dataset,
        total_clients=total_clients,
        alpha=alpha
    )

    for i in range(total_clients):
        paths_i = client_paths[i]
        targets_i = client_targets[i]
        domains_i = client_domains[i]

        # Calcul des proportions par domaine
        total_samples = len(domains_i)
        domain_counts = [domains_i.count(d_idx) for d_idx in range(len(DOMAINS))]
        domain_props = [count / total_samples for count in domain_counts]
        
        # Format d'affichage : "photo: 45.2%, art: 10.0%, ..."
        props_str = ", ".join([f"{DOMAINS[d]}: {p*100:.1f}%" for d, p in enumerate(domain_props)])

        # Split train/test
        dataset_size = total_samples
        indices = list(range(dataset_size))
        random.seed(0)
        random.shuffle(indices)

        split_idx = int(0.8 * dataset_size)
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]

        train_data = np.array([transform(default_loader(paths_i[idx])) for idx in train_idx])
        train_targets = np.array([targets_i[idx] for idx in train_idx])

        test_data = np.array([transform(default_loader(paths_i[idx])) for idx in test_idx])
        test_targets = np.array([targets_i[idx] for idx in test_idx])

        train_subset = torch.utils.data.TensorDataset(
            torch.from_numpy(train_data),
            torch.from_numpy(train_targets)
        )
        test_subset = torch.utils.data.TensorDataset(
            torch.from_numpy(test_data),
            torch.from_numpy(test_targets)
        )

        print(f"Client {i:02d} ({total_samples} samples) | Proportions: [{props_str}]")
        print(f"  └── Train: {len(train_subset)}, Test: {len(test_subset)}")

        torch.save(train_subset, f"/tmp/app/data/train/train_subset-{i}.pth")
        torch.save(test_subset, f"/tmp/app/data/test/test_subset-{i}.pth")