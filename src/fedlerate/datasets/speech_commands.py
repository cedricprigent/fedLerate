import os
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, TensorDataset

ALLOWED_CLASSES = ['yes', 'no', 'up', 'down']
class_to_idx = {cls: i for i, cls in enumerate(sorted(ALLOWED_CLASSES))}

def normalize_waveform(waveform):
    # Scale to [-1, 1] range (Standard for audio)
    # This is critical because the original files vary in volume.
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
    return waveform

def load_sample(path, class_to_idx):
    waveform, sample_rate = torchaudio.load(path)
    label = os.path.basename(os.path.dirname(path))
    
    # Ensure waveform is 16000 samples (1 second at 16kHz)
    if waveform.size(1) > 16000:
        waveform = waveform[:, :16000]
    elif waveform.size(1) < 16000:
        waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.size(1)))

    waveform = normalize_waveform(waveform.squeeze(0))
    return waveform, class_to_idx[label]

def generate_speech_commands_partitions(alpha, n_clients, root='~/datasets'):
    """
    Partitions the Speech Commands dataset such that each client has data from exactly ONE speaker.
    The 'alpha' parameter is ignored in this mode as assignments are 1-to-1 based on speaker IDs.
    """
    root = os.path.expanduser(root)
    np.random.seed(0) # For reproducibility in splitting speaker data
    
    # 1. Collect all samples from all subsets
    print("Loading Speech Commands dataset structure...")
    all_files = []
    for subset in ['training', 'validation', 'testing']:
        try:
            ds = torchaudio.datasets.SPEECHCOMMANDS(root=root, subset=subset, download=True)
            all_files.extend(ds._walker)
        except Exception as e:
            print(f"Warning: Could not load subset {subset}: {e}")
    
    if not all_files:
        raise RuntimeError(f"No Speech Commands data found in {root}")

    if ALLOWED_CLASSES:
        all_files = [p for p in all_files if os.path.basename(os.path.dirname(p)) in ALLOWED_CLASSES]

    # 2. Get unique classes and map them
    subset_classes = sorted(list(set(os.path.basename(os.path.dirname(p)) for p in all_files)))
    class_to_idx = {cls: i for i, cls in enumerate(subset_classes)}
    
    # 3. Group by speaker
    speaker_to_files = {}
    for p in all_files:
        speaker_id = os.path.basename(p).split('_nohash_')[0]
        if speaker_id not in speaker_to_files:
            speaker_to_files[speaker_id] = []
        speaker_to_files[speaker_id].append(p)
    
    # 4. Filter and sort speakers
    # We want speakers with enough samples to split into train/test
    MIN_SAMPLES = 17
    valid_speakers = sorted([sid for sid, files in speaker_to_files.items() if len(files) >= MIN_SAMPLES])
    
    if len(valid_speakers) < n_clients:
        print(f"Warning: Only {len(valid_speakers)} speakers available with >= {MIN_SAMPLES} samples, but requested {n_clients} clients.")
        n_clients = len(valid_speakers)
    
    print(f"Partitioning for {n_clients} clients (one speaker per client)...")
    
    os.makedirs('/tmp/app/data/train', exist_ok=True)
    os.makedirs('/tmp/app/data/test', exist_ok=True)
    
    for i in range(n_clients):
        sid = valid_speakers[i]
        files = speaker_to_files[sid]
        np.random.shuffle(files)
        
        # Split speaker's data into 80% train, 20% test
        split_idx = max(1, int(0.8 * len(files)))
        train_files = files[:split_idx]
        test_files = files[split_idx:]
        
        # Helper to process and save
        def save_subset(file_list, path):
            data, targets = [], []
            for p in file_list:
                # Ensure the file belongs to our subset
                label = os.path.basename(os.path.dirname(p))
                if label in class_to_idx:
                    x, y = load_sample(p, class_to_idx)
                    data.append(x)
                    targets.append(y)
            if data:
                torch.save(TensorDataset(torch.stack(data), torch.tensor(targets)), path)

        save_subset(train_files, f"/tmp/app/data/train/train_subset-{i}.pth")
        save_subset(test_files, f"/tmp/app/data/test/test_subset-{i}.pth")
        
        if i % 10 == 0 or i == n_clients - 1:
            print(f"Progress: {i+1}/{n_clients} clients created.")

    print(f"Successfully created {n_clients} partitions in /tmp/app/data/")
