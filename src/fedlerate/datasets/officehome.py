import os
import numpy as np
import torch
from collections import Counter
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose, ToTensor, Resize
from fedlerate.data.transforms import get_normalization


# ==============================================================================
# 1. CALCUL DU TOP 15 DES CLASSES ET MAPPING
# ==============================================================================
def get_top_k_classes_mapping(raw_dir, top_k=15):
    """Calcule les K classes les plus fréquentes et crée la réindexation [0, K-1]."""
    domains = ["Art", "Clipart", "Product", "Real World"]
    all_targets = []

    for d in domains:
        ds = ImageFolder(os.path.join(raw_dir, d))
        all_targets.extend([target for _, target in ds.samples])

    class_counts = Counter(all_targets)
    top_classes = [cls for cls, _ in class_counts.most_common(top_k)]
    class_mapping = {old_cls: new_idx for new_idx, old_cls in enumerate(top_classes)}
    
    print(f"--> Top {top_k} classes conservées : {top_classes}")
    return class_mapping


# ==============================================================================
# 2. GÉNÉRATION STRICTE (1 Domaine par Client + Top 15 Classes)
# ==============================================================================
def generate_office_home_strict(
    raw_dir="/tmp/app/data/office_home_raw", 
    output_dir="/tmp/app/data", 
    n_clients_per_domain=10, 
    top_k=15
):
    
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    domains = ["Art", "Clipart", "Product", "Real World"]
    transform = Compose([ToTensor(), Resize(size=(64, 64), antialias=True), get_normalization(channels=3)])
    
    # 1. Récupération du mapping des 15 classes majoritaires
    class_mapping = get_top_k_classes_mapping(raw_dir, top_k=top_k)

    print(f"\n[Scenario Strict - Top {top_k}] Génération ({n_clients_per_domain * 4} clients au total)...")

    global_client_idx = 0

    # 2. Traitement séparé domaine par domaine
    for domain_idx, domain_name in enumerate(domains):
        domain_path = os.path.join(raw_dir, domain_name)
        ds = ImageFolder(domain_path)

        # Filtrer uniquement les échantillons appartenant au Top 15
        filtered_samples = [
            (path, class_mapping[target]) 
            for path, target in ds.samples 
            if target in class_mapping
        ]

        print(f"--> Domaine '{domain_name}' : {len(filtered_samples)} images conservées.")

        # Mélange des indices du domaine
        np.random.seed(42 + domain_idx)
        indices = np.arange(len(filtered_samples))
        np.random.shuffle(indices)

        # Découpage équitable entre les clients réservés à ce domaine
        client_splits = np.array_split(indices, n_clients_per_domain)

        for split in client_splits:
            client_samples = [filtered_samples[idx] for idx in split]

            # Split Train (80%) / Test (20%) pour ce client
            n_train = int(0.8 * len(client_samples))
            train_files = client_samples[:n_train]
            test_files = client_samples[n_train:]

            # Chargement et transformation des images en tenseurs PyTorch
            train_x = torch.stack([transform(default_loader(p[0])) for p in train_files])
            train_y = torch.tensor([p[1] for p in train_files], dtype=torch.long)

            test_x = torch.stack([transform(default_loader(p[0])) for p in test_files])
            test_y = torch.tensor([p[1] for p in test_files], dtype=torch.long)

            # Sauvegarde des fichiers .pth clients
            torch.save(
                torch.utils.data.TensorDataset(train_x, train_y), 
                f"{train_dir}/train_subset-{global_client_idx}.pth"
            )
            torch.save(
                torch.utils.data.TensorDataset(test_x, test_y), 
                f"{test_dir}/test_subset-{global_client_idx}.pth"
            )
            
            global_client_idx += 1

    print(f"\n--> Partitionnement strict terminé. {global_client_idx} clients sauvegardés dans : {output_dir}")
