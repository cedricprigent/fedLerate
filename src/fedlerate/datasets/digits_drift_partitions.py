from torchvision import datasets, transforms
from datasets import load_dataset
from typing import List, Tuple
import torch
import os
from torch.utils.data import TensorDataset, random_split, DataLoader
from PIL import Image # Required for ImageFolder default_loader
from fedlerate.datasets.pacs_soft_partitions import dirichlet_partitions

def create_digits_drift_partitions(
    alpha: float, # Not directly used for Dirichlet distribution in this specific partitioning scheme
    n_clients: int,
    data_dir: str = "/tmp/app/data"
) -> None:
    """
    Create federated digit drift partitions from MNIST, SVHN, and MNIST-M datasets.
    Clients are distributed across these three domains.
    
    Args:
        alpha: Concentration parameter (not used for Dirichlet in this function, but for CLI consistency).
        n_clients: Total number of federated clients.
        data_dir: Directory to download datasets and where MNIST-M is expected.
    """
    os.makedirs(os.path.join(data_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'test'), exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("#################################################")
    print("# Generating partitions for Digits Drift\t#")
    print(f"# - alpha = {alpha}\t\t\t\t#")
    print(f"# - total clients = {n_clients}\t\t\t\t#")
    print("#################################################")
    
    # Load datasets
    mnist_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    svhn_train = datasets.SVHN(data_dir, split='train', download=True, transform=transform)
    svhn_test = datasets.SVHN(data_dir, split='test', download=True, transform=transform)

    mnist_m = load_dataset("Mike0307/MNIST-M", cache_dir=data_dir)
    mnistm_train, mnistm_test = mnist_m["train"], mnist_m["test"]

    dataset_names = ["MNIST", "SVHN", "MNIST-M"]

    all_train_datasets = [mnist_train, svhn_train, mnistm_train]
    all_test_datasets = [mnist_test, svhn_test, mnistm_test]
    
    train_subsets = [None] * n_clients
    test_subsets = [None] * n_clients

    # Distribute clients as evenly as possible across the three datasets
    clients_per_dataset = [n_clients // len(all_train_datasets)] * len(all_train_datasets)
    for i in range(n_clients % len(all_train_datasets)):
        clients_per_dataset[i] += 1

    client_idx_counter = 0
    for ds_idx, (train_ds, test_ds) in enumerate(zip(all_train_datasets, all_test_datasets)):
        num_clients_for_this_ds = clients_per_dataset[ds_idx]
        print(f"Assigning {num_clients_for_this_ds} clients to {dataset_names[ds_idx]}")

        if num_clients_for_this_ds == 0:
            continue

        # Helper to convert any dataset to TensorDataset for consistent splitting and saving
        def to_tensor_dataset(dataset):
            data_list = []
            targets_list = []
            for i in range(len(dataset)):
                img, label = dataset[i]
                data_list.append(img)
                targets_list.append(label)
            return TensorDataset(torch.stack(data_list), torch.tensor(targets_list))

        full_train_tensor_ds = to_tensor_dataset(train_ds)
        full_test_tensor_ds = to_tensor_dataset(test_ds)

        # Split the full dataset for this domain into `num_clients_for_this_ds` parts
        train_lengths = [len(full_train_tensor_ds) // num_clients_for_this_ds] * num_clients_for_this_ds
        for i in range(len(full_train_tensor_ds) % num_clients_for_this_ds):
            train_lengths[i] += 1
        
        test_lengths = [len(full_test_tensor_ds) // num_clients_for_this_ds] * num_clients_for_this_ds
        for i in range(len(full_test_tensor_ds) % num_clients_for_this_ds):
            test_lengths[i] += 1

        split_train_ds = random_split(full_train_tensor_ds, train_lengths)
        split_test_ds = random_split(full_test_tensor_ds, test_lengths)

        for i in range(num_clients_for_this_ds):
            train_subsets[client_idx_counter] = split_train_ds[i]
            test_subsets[client_idx_counter] = split_test_ds[i]
            client_idx_counter += 1

    # Save subsets
    for i, (train_subset, test_subset) in enumerate(zip(train_subsets, test_subsets)):
        if train_subset is not None and test_subset is not None: # Ensure subset was assigned
            print(f"subset {i}: {len(train_subset)} train data, {len(test_subset)} test data")
            torch.save(train_subset, os.path.join(data_dir, f"train/train_subset-{i}.pth"))
            torch.save(test_subset, os.path.join(data_dir, f"test/test_subset-{i}.pth"))
        else:
            print(f"subset {i}: No data assigned (this should not happen if n_clients > 0)")

    train_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in all_train_datasets]
    test_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in all_test_datasets]

    for idx, (test_loader, train_loader) in enumerate(zip(test_loaders, train_loaders)):
        n_partitions = clients_per_dataset[idx]
        
        train_partitions = dirichlet_partitions(dataset=train_loader.dataset, num_clients=n_partitions, alpha=alpha, transform=transform)
        test_partitions = dirichlet_partitions(dataset=test_loader.dataset, num_clients=n_partitions, alpha=alpha, transform=transform)
        
        train_subsets = []
        test_subsets = []
        for train_partition, test_partition in zip(train_partitions, test_partitions):

            train = torch.utils.data.TensorDataset(
                torch.from_numpy(train_partition.data),
                torch.from_numpy(train_partition.targets)
            )

            test = torch.utils.data.TensorDataset(
                torch.from_numpy(test_partition.data),
                torch.from_numpy(test_partition.targets)
            )
            train_subsets.append(train)
            test_subsets.append(test)
        
        for i, (train_subset, test_subset) in enumerate(zip(train_subsets, test_subsets)):
                print(f"subset {i + idx * n_partitions}: {len(train_subset)} train data, {len(test_subset)} test data")
        
                torch.save(train_subset, f"/tmp/app/data/train/train_subset-{i + idx * n_partitions}.pth")
                torch.save(test_subset, f"/tmp/app/data/test/test_subset-{i + idx * n_partitions}.pth")