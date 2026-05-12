from torchvision import datasets, transforms
from datasets import load_dataset
import torch
import os
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from datasets import Dataset as HFDataset

def create_digits_drift_partitions(
    alpha: float, 
    n_clients: int,
    data_dir: str = "/tmp/app/data"
) -> None:
    
    os.makedirs(os.path.join(data_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'test'), exist_ok=True)

    # Note: Normalize with 3 values for RGB compatibility
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Lambda(lambda x: x.convert("RGB") if hasattr(x, "convert") else x),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    print("#################################################")
    print("# Generating partitions for Digits Drift\t#")
    print(f"# - total clients = {n_clients}\t\t\t\t#")
    print("#################################################")
    
    # 1. Load original datasets (Lazy/Compressed)
    mnist_train = datasets.MNIST(data_dir, train=True, download=True)
    mnist_test = datasets.MNIST(data_dir, train=False, download=True)

    svhn_train = datasets.SVHN(data_dir, split='train', download=True)
    svhn_test = datasets.SVHN(data_dir, split='test', download=True)

    mnist_m = load_dataset("Mike0307/MNIST-M", cache_dir=data_dir)
    mnistm_train, mnistm_test = mnist_m["train"], mnist_m["test"]

    dataset_names = ["MNIST-M", "MNIST", "SVHN"]
    all_train_datasets = [mnistm_train, mnist_train, svhn_train]
    all_test_datasets = [mnistm_test, mnist_test, svhn_test]

    # 2. Distribute client counts
    clients_per_dataset = [n_clients // len(all_train_datasets)] * len(all_train_datasets)
    for i in range(n_clients % len(all_train_datasets)):
        clients_per_dataset[i] += 1

    # Helper function to convert a specific slice of indices to a TensorDataset
    def get_subset_as_tensor(full_ds, indices, transform):
        data_list = []
        targets_list = []
        for idx in indices:
            item = full_ds[int(idx)]
            # HF vs Torchvision check
            if isinstance(item, dict):
                img, label = item["image"], item["label"]
            else:
                img, label = item
            
            # Ensure PIL for transform
            if not isinstance(img, torch.Tensor):
                if hasattr(img, "convert"):
                    img = img.convert("RGB")
            
            data_list.append(transform(img))
            targets_list.append(label)
            
        return TensorDataset(torch.stack(data_list), torch.tensor(targets_list))

    client_global_idx = 0

    # 3. Process Domain by Domain, Client by Client
    for ds_idx, (train_ds, test_ds) in enumerate(zip(all_train_datasets, all_test_datasets)):
        n_clients_here = clients_per_dataset[ds_idx]
        if n_clients_here == 0: continue
        
        print(f"Processing {dataset_names[ds_idx]} for {n_clients_here} clients...")

        # Shuffle indices for this domain
        train_idxs = np.random.permutation(len(train_ds))
        test_idxs = np.random.permutation(len(test_ds))

        # Split indices
        train_splits = np.array_split(train_idxs, n_clients_here)
        test_splits = np.array_split(test_idxs, n_clients_here)

        for c_local_idx in range(n_clients_here):
            # Create and Save Train Subset
            train_tensor_ds = get_subset_as_tensor(train_ds, train_splits[c_local_idx], transform)
            torch.save(train_tensor_ds, os.path.join(data_dir, f"train/train_subset-{client_global_idx}.pth"))
            
            # Create and Save Test Subset
            test_tensor_ds = get_subset_as_tensor(test_ds, test_splits[c_local_idx], transform)
            torch.save(test_tensor_ds, os.path.join(data_dir, f"test/test_subset-{client_global_idx}.pth"))
            
            print(f" Saved client {client_global_idx} ({dataset_names[ds_idx]}): {len(train_tensor_ds)} train samples")
            
            # CRITICAL: Clean up memory after each client
            del train_tensor_ds
            del test_tensor_ds
            
            client_global_idx += 1

    print("\nAll partitions created successfully.")