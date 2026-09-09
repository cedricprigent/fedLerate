import time

from fedlerate.models.models import PFL_AutoEncoder, TransformerEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import random
import os
import torchaudio


DEVICE='cuda' if torch.cuda.is_available() else 'cpu'


def get_background_noise_list(root='~/datasets', sample_rate=16000):
    root = os.path.expanduser(root)
    # The noise files are in the _background_noise_ folder
    noise_path = os.path.join(root, 'SpeechCommands/speech_commands_v0.02/_background_noise_')
    
    noise_list = []
    
    # Supported audio formats
    valid_extensions = ('.wav', '.mp3')
    
    if not os.path.exists(noise_path):
        print(f"Warning: Background noise folder not found at {noise_path}")
        return []

    for filename in os.listdir(noise_path):
        if filename.endswith(valid_extensions):
            file_path = os.path.join(noise_path, filename)
            # Load waveform
            waveform, sr = torchaudio.load(file_path)
            
            # Resample if necessary to match your model's requirement
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                waveform = resampler(waveform)
                
            noise_list.append(waveform)

    return noise_list


class BackgroundNoiseMixer:
    def __init__(self, noise_dataset, snr_range=(5, 15)):
        """
        noise_dataset: list of waveforms (tensors)
        snr_range: range of Signal-to-Noise Ratio (dB)
        """
        self.noise_dataset = noise_dataset
        self.snr_range = snr_range

    def __call__(self, waveform):
        # 1. Randomly pick a noise sample
        noise = random.choice(self.noise_dataset)
        
        # 2. Extract a random snippet of the same length as the waveform
        start = random.randint(0, noise.size(1) - waveform.size(1))
        snippet = noise[:, start : start + waveform.size(1)]
        
        # 3. Mix with random SNR
        snr = random.uniform(*self.snr_range)
        
        # Calculate powers
        waveform_power = waveform.pow(2).mean()
        noise_power = snippet.pow(2).mean()
        
        # Scale noise to match SNR
        scale = (waveform_power / (noise_power * (10**(snr / 10)))).sqrt()
        
        # Add to original and normalize back
        mixed = waveform + (snippet * scale)
        return mixed / mixed.abs().max()


def spectral_augmentation(x):
    # Convert to Frequency Domain
    x_freq = torch.fft.rfft(x)
    # Randomly zero out high frequencies (noise)
    mask = torch.rand(x_freq.shape) > 0.1 
    x_freq = x_freq * mask.to(x_freq.device)
    # Back to Time Domain
    return torch.fft.irfft(x_freq, n=x.shape[-1])

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
    

def get_client_loss_function(y_train, num_classes=5):
    # Compute weights based on this client's specific labels
    if isinstance(y_train, torch.Tensor):
        labels_np = y_train.cpu().detach().numpy().flatten()
    elif isinstance(y_train, list):
        # Flatten if it's a list of lists or mixed types
        labels_np = np.array(y_train).flatten()
    else:
        labels_np = np.array(y_train).flatten()
    
    # Filter labels to only include those within [0, num_classes)
    mask = (labels_np >= 0) & (labels_np < num_classes)
    labels_np = labels_np[mask]
    
    if len(labels_np) == 0:
        return nn.CrossEntropyLoss()

    classes_present = np.unique(labels_np)
    weights_present = compute_class_weight(
        class_weight='balanced', 
        classes=classes_present, 
        y=labels_np
    )
    weights_present = np.array([0.272, 0.728])
    
    # Initialize weights with 1.0 for all classes
    full_weights = np.ones(num_classes, dtype=np.float32)
    for cls, w in zip(classes_present, weights_present):
        full_weights[cls] = w
        
    # Convert to tensor and move to device
    class_weights = torch.tensor(full_weights, dtype=torch.float32).to(DEVICE)
    return nn.CrossEntropyLoss(weight=class_weights)


def train(model, dataloader, epochs, lr=3e-4, device=DEVICE, noise_mixer=None):
    if isinstance(model, TransformerEncoder):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # The Transformer often benefits from a simple CrossEntropyLoss 
        # if you are using labels, or the Deep SAD loss if unsupervised.
        # if hasattr(dataloader.dataset, 'tensors'):
        #     criterion = get_client_loss_function(dataloader.dataset.tensors[1], num_classes=2)
        # else:
        #     criterion = get_client_loss_function(dataloader.dataset.targets, num_classes=2)

        criterion = FocalLoss(alpha=0.25, gamma=0.5)
        
        model.train()
        for epoch in range(epochs):
            for batch in dataloader:
                x, y = batch[0].to(device), batch[1].to(device)
                
                # 1. Apply Spectral Augmentation
                x = spectral_augmentation(x)
                
                # 2. Forward pass
                logits = model(x)
                
                # Filter out invalid labels (-1) and ensure labels are within the number of classes
                mask = (y >= 0) & (y < logits.size(1))
                logits = logits[mask]
                y = y[mask]
                
                # 3. Loss Calculation
                # Using CrossEntropy if using labels (Supervised/Semi-supervised)
                loss = criterion(logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return loss.item(), 0.0
    

    if isinstance(model, PFL_AutoEncoder):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()

        for epoch in range(epochs):
            train_loss = 0
            for batch, x in enumerate(dataloader):
                if isinstance(x, (list, tuple)): x = x[0]
                x = x.to(device)
                # Forward pass
                output = model(x)

                # Calculate loss
                loss = criterion(output, x)
                train_loss += loss.item()

                # Zero grad
                optimizer.zero_grad()
                
                # Backprog
                loss.backward()

                # Optimize
                optimizer.step()

        train_loss = train_loss / len(dataloader)
        return train_loss, 0.0

    else:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()

        for epoch in range(epochs):
            train_loss = 0
            train_accuracy = 0

            for batch, (x, y) in enumerate(dataloader):
                if noise_mixer is not None:
                    if random.random() < 0.52:  # 50% chance to apply noise augmentation
                        x = noise_mixer(x)  # Apply background noise augmentation

                if len(x.shape) == 3:
                    x = x.unsqueeze(1)  # Add channel dimension for MNIST

                x = x.to(device)
                y = y.to(device)

                # Forward pass
                output = model(x)

                # Ensure y is within bounds [0, output.size(1)-1]
                if torch.any((y < 0) | (y >= output.size(1))):
                     raise ValueError(f"Label out of bounds: min={y.min()}, max={y.max()}, num_classes={output.size(1)}")

                # Calculate loss
                loss = criterion(output, y)
                train_loss += loss.item()
                train_accuracy += accuracy_fn(y, torch.argmax(output, dim=1))

                # Zero grad
                optimizer.zero_grad()
                
                # Backprog
                loss.backward()

                # Optimize
                optimizer.step()

        train_loss = train_loss / len(dataloader)
        train_accuracy = train_accuracy / len(dataloader)
        return train_loss, train_accuracy


def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """

    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def train_fedprox(model, dataloader, epochs, mu=0.01, lr=3e-4, device=DEVICE, noise_mixer=None):
    if isinstance(model, PFL_AutoEncoder):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        global_model = copy.deepcopy(model)
        model.train()

        for epoch in range(epochs):
            train_loss = 0

            for batch, x in enumerate(dataloader):
                if isinstance(x, (list, tuple)): x = x[0]
                x = x.to(device)
                # Forward pass
                output = model(x)

                # Calculate loss
                loss = criterion(output, x) + proximal_term(model, global_model, mu=mu)
                train_loss += loss.item()

                # Zero grad
                optimizer.zero_grad()

                # Backprog
                loss.backward()

                # Optimize
                optimizer.step()

        train_loss = train_loss / len(dataloader)
        return train_loss, 0.0

    else:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        global_model = copy.deepcopy(model)

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            correct_preds = 0
            total_samples = 0

            for x, y in dataloader:
                if noise_mixer is not None and random.random() < 0.52:
                    x = noise_mixer(x)

                if len(x.shape) == 3:
                    x = x.unsqueeze(1)

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # 1. Forward pass & task loss
                output = model(x)
                loss = criterion(output, y)

                # 2. Backward pass builds standard task gradients
                loss.backward()

                # 3. Add proximal gradient: \mu * (w_local - w_global)
                with torch.no_grad():
                    for p_local, p_global in zip(model.parameters(), global_model.parameters()):
                        if p_local.grad is not None:
                            p_local.grad.add_(p_local - p_global, alpha=mu)

                # 4. Step optimizer with combined gradients
                optimizer.step()

                # Non-blocking tracking
                total_loss += loss.detach() * x.size(0)
                correct_preds += (output.argmax(dim=1) == y).sum()
                total_samples += x.size(0)

            train_loss = (total_loss / total_samples).item()
            train_accuracy = (correct_preds / total_samples).item()
        return train_loss, train_accuracy


def proximal_term(model, global_model, mu):
    """Compute the FedProx proximal term."""
    proximal_term = 0.0
    for param, global_param in zip(model.parameters(), global_model.parameters()):
        proximal_term += (param - global_param).norm(2)
    return (mu / 2) * proximal_term


def train_ditto(personalized_model, global_model, dataloader, epochs, lr=3e-4, lambda_=0.5, device=DEVICE, noise_mixer=None):
    if isinstance(personalized_model, PFL_AutoEncoder):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(personalized_model.parameters(), lr=lr)

        personalized_model.train()

        for epoch in range(epochs):
            train_loss = 0

            for batch, x in enumerate(dataloader):
                if isinstance(x, (list, tuple)): x = x[0]
                x = x.to(device)
                # Forward pass
                output = personalized_model(x)

                # Calculate loss
                prox = 0
                for param, global_param in zip(personalized_model.parameters(), global_model.parameters()):
                    prox += (param - global_param).norm(2)
                loss = criterion(output, x) + lambda_ * prox
                train_loss += loss.item()

                # Zero grad
                optimizer.zero_grad()

                # Backprog
                loss.backward()

                # Optimize
                optimizer.step()

        train_loss = train_loss / len(dataloader)
        return train_loss, 0.0
    else:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(personalized_model.parameters(), lr=lr)

        personalized_model.train()

        for epoch in range(epochs):
            train_loss = 0
            train_accuracy = 0

            for batch, (x, y) in enumerate(dataloader):
                if noise_mixer is not None:
                    if random.random() < 0.52:  # 50% chance to apply noise augmentation
                        x = noise_mixer(x)  # Apply background noise augmentation

                if len(x.shape) == 3:
                    x = x.unsqueeze(1)  # Add channel dimension for MNIST

                x = x.to(device)
                y = y.to(device)

                # Forward pass
                output = personalized_model(x)

                # Calculate loss
                prox = 0
                for param, global_param in zip(personalized_model.parameters(), global_model.parameters()):
                    prox += (param - global_param).norm(2)
                loss = criterion(output, y) + lambda_ * prox
                train_loss += loss.item()
                train_accuracy += accuracy_fn(y, torch.argmax(output, dim=1))

                # Zero grad
                optimizer.zero_grad()

                # Backprog
                loss.backward()

                # Optimize
                optimizer.step()

        train_loss = train_loss / len(dataloader)
        train_accuracy = train_accuracy / len(dataloader)
        return train_loss, train_accuracy


def train_fedmix(models, weights, dataloader, epochs, lr=3e-4, device=DEVICE, noise_mixer=None, budget=1.0):
    if isinstance(models[0], PFL_AutoEncoder):
        criterion = torch.nn.MSELoss()
        params = [weights]
        for model in models:
            params += list(model.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)

        epoch_training_budget = len(dataloader) * budget

        for epoch in range(epochs):
            train_loss = 0

            for batch, x in enumerate(dataloader):
                if batch >= epoch_training_budget:
                    break  # Limit training to the budget for this epoch

                if isinstance(x, (list, tuple)): x = x[0]
                x = x.to(device)
                # Forward pass
                normalized_weights = torch.softmax(weights, dim=0)
                model_outputs = torch.stack([m(x) for m in models])  # (k, batch, features)
                output = (normalized_weights.view(-1, 1, 1) * model_outputs).sum(0)

                # Calculate loss
                loss = criterion(output, x)
                train_loss += loss.item()

                # Zero grad
                optimizer.zero_grad()

                # Backprog
                loss.backward()

                # Optimize
                optimizer.step()

        train_loss = train_loss / len(dataloader)
        return train_loss, 0.0
    else:
        criterion = torch.nn.CrossEntropyLoss()
        params = [weights]
        for model in models:
            params += list(model.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)

        k = len(models)
        epoch_training_budget = len(dataloader) / k

        for epoch in range(epochs):
            train_loss = 0
            train_accuracy = 0

            for batch, (x, y) in enumerate(dataloader):
                if batch >= epoch_training_budget:
                    break  # Limit training to the budget for this epoch

                if noise_mixer is not None:
                    if random.random() < 0.52:  # 50% chance to apply noise augmentation
                        x = noise_mixer(x)  # Apply background noise augmentation

                if len(x.shape) == 3:
                    x = x.unsqueeze(1)  # Add channel dimension for MNIST

                x = x.to(device)
                y = y.to(device)

                # Forward pass
                normalized_weights = torch.softmax(weights, dim=0)

                model_outputs = torch.stack([m(x) for m in models])  # (k, batch, classes)

                output = (normalized_weights.view(-1, 1, 1) * model_outputs).sum(0)


                # Calculate loss
                loss = criterion(output, y)
                train_loss += loss.item()
                train_accuracy += accuracy_fn(y, torch.argmax(output, dim=1))

                # Zero grad
                optimizer.zero_grad()
                
                # Backprog
                loss.backward()

                # Optimize
                optimizer.step()

        train_loss = train_loss / len(dataloader)
        train_accuracy = train_accuracy / len(dataloader)
        return train_loss, train_accuracy



def evaluate(model, dataloader, device=DEVICE):
    if isinstance(model, TransformerEncoder):
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch[0].to(device), batch[1].to(device)
                
                # Forward pass
                logits = model(x)
                
                # Get predictions (Logits -> Class)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        return acc

    model.eval()
    if isinstance(model, PFL_AutoEncoder):
        errs = []
        labels = []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1]
                else:
                    x = batch
                    y = torch.zeros(x.size(0))

                x = x.to(device)
                recon = model(x)
                
                # --- FIX: Ensure 3D shape ---
                # If x is [Batch, Features], reshape it to [Batch, 1, Features]
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                    recon = recon.unsqueeze(1)
                
                # Now the mean calculation over dim (1, 2) is valid
                batch_errs = ((x - recon) ** 2).mean(dim=(1, 2))
                
                errs.append(batch_errs.cpu())
                labels.append(y.cpu())

        errs = torch.cat(errs).numpy()
        threshold = np.percentile(errs, 98)
        
        # 1. Convert to numpy array
        labels = torch.cat(labels).numpy()

        binary_labels = (labels != 0).astype(int)
        
        # Now preds is a 1D array of length Batch
        preds = (errs > threshold).astype(int)
        
        f1 = f1_score(binary_labels, preds, zero_division=0.0)
        # auc = roc_auc_score(labels, errs)

        return f1
    else:
        correct = 0
        total = 0

        with torch.no_grad():
            for batch, (x, y) in enumerate(dataloader):
                x = x.to(device)
                if len(x.shape) == 3:
                    x = x.unsqueeze(1)  # Add channel dimension for MNIST
                y = y.to(device)

                # Forward pass
                output = model(x)

                # Compute score
                _, predicted = torch.max(output, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        return correct / total


def evaluate_erm(model, dataloader, sampling_ratio=1.0, device=DEVICE):
    # Evaluate the model on the given dataloader with a sampling ratio
    correct = 0
    total = 0

    n_batches = len(dataloader)
    n_sampled_batches = max(1, int(n_batches * sampling_ratio))

    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            if batch >= n_sampled_batches:
                break
            
            x = x.to(device)
            if len(x.shape) == 3:
                x = x.unsqueeze(1)  # Add channel dimension for MNIST
            y = y.to(device)

            # Forward pass
            output = model(x)

            # Compute score
            _, predicted = torch.max(output, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return correct / total


def evaluate_fedmix(models, weights, dataloader, device=DEVICE, threshold=0.05):
    for model in models:
        model.eval()

    if isinstance(models[0], PFL_AutoEncoder):
        errs = []
        labels = []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1]
                else:
                    x = batch
                    y = torch.zeros(x.size(0))
                
                x = x.to(device)
                normalized_weights = torch.softmax(weights, dim=0)
                model_outputs = torch.stack([m(x) for m in models])  # (k, batch, features)
                recon = (normalized_weights.view(-1, 1, 1) * model_outputs).sum(0)
                errs.append(((x - recon) ** 2).mean(dim=1))
                labels.append(y.cpu())

        errs = torch.cat(errs).cpu().numpy()
        labels = torch.cat(labels).numpy()

        binary_labels = (labels != 0).astype(int)
        
        preds = (errs > threshold).astype(int)
        f1 = f1_score(binary_labels, preds, zero_division=0.0)

        return f1

    else:
        correct = 0
        total = 0

        normalized_weights = torch.softmax(weights, dim=0).to(device)

        with torch.no_grad():
            for batch, (x, y) in enumerate(dataloader):
                x = x.to(device)
                if len(x.shape) == 3:
                    x = x.unsqueeze(1)  # Add channel dimension for MNIST
                y = y.to(device)

                # Forward pass
                model_outputs = torch.stack([m(x) for m in models])  # (k, batch, classes)

                output = (normalized_weights.view(-1, 1, 1) * model_outputs).sum(0)

                # Compute score
                _, predicted = torch.max(output, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        return correct / total