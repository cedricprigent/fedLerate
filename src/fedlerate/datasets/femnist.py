import os
import torch
import numpy as np
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import TensorDataset
from fedlerate.data.transforms import get_normalization


def create_femnist_partitions(
    n_clients: int,
    test_ratio: float = 0.2,
    data_dir: str = "/tmp/app/data"
) -> None:
    """
    Creates train/test partitions for FEMNIST from flwrlabs/femnist and saves them as TensorDatasets (.pth).
    
    :param n_clients: Number of FL clients to generate.
    :param test_ratio: Percentage of data reserved for local test set per client.
    :param data_dir: Target output folder path.
    """
    os.makedirs(os.path.join(data_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'test'), exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Lambda(lambda x: x.convert("RGB") if hasattr(x, "convert") else x),
        transforms.ToTensor(),
        get_normalization(channels=3)
    ])
    
    print("#################################################")
    print("# Generating partitions for FEMNIST\t\t#")
    print(f"# - total clients = {n_clients}\t\t\t\t#")
    print("#################################################")

    # 1. Load HuggingFace FEMNIST dataset (single 'train' split)
    femnist_raw = load_dataset("flwrlabs/femnist", cache_dir=data_dir)["train"]

    def process_subset_indices(full_ds, indices):
        """Converts index subsets to PyTorch TensorDataset."""
        data_list, targets_list = [], []
        for idx in indices:
            item = full_ds[int(idx)]
            img = item["image"]
            label = item["character"]

            if hasattr(img, "convert"):
                img = img.convert("RGB")

            data_list.append(transform(img))
            targets_list.append(label)

        return TensorDataset(torch.stack(data_list), torch.tensor(targets_list))

    # Map sample indices by unique writer ID
    print("Grouping samples by writer ID...")
    writer_to_indices = {}
    for idx, writer in enumerate(femnist_raw["writer_id"]):
        if writer not in writer_to_indices:
            writer_to_indices[writer] = []
        writer_to_indices[writer].append(idx)

    # Select unique writers for the requested n_clients
    all_writers = list(writer_to_indices.keys())
    np.random.shuffle(all_writers)
    selected_writers = all_writers[:n_clients]

    for client_idx, writer_id in enumerate(selected_writers):
        indices = writer_to_indices[writer_id]
        np.random.shuffle(indices)

        # Local train/test split per writer
        split_point = int(len(indices) * (1 - test_ratio))
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]

        train_tensor_ds = process_subset_indices(femnist_raw, train_indices)
        test_tensor_ds = process_subset_indices(femnist_raw, test_indices)

        torch.save(train_tensor_ds, os.path.join(data_dir, f"train/train_subset-{client_idx}.pth"))
        torch.save(test_tensor_ds, os.path.join(data_dir, f"test/test_subset-{client_idx}.pth"))

        print(f" Saved client {client_idx} (Writer {writer_id}): {len(train_tensor_ds)} train, {len(test_tensor_ds)} test samples")

        del train_tensor_ds, test_tensor_ds

    print("\nFEMNIST partitions created successfully.")