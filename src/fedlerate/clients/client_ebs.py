import numpy as np
import torch
from fedlerate.clients.client_base import BaseClient
from fedlerate.models.functions import evaluate_erm


class ClientEBS(BaseClient):
    def __init__(self, client_id, config, device):
        super(ClientEBS, self).__init__(client_id, config, device)

    def select_best_cluster(self, cluster_models):
        """
        Evaluates local empirical performance on all m candidate cluster 
        models via forward passes and selects the best cluster index.
        """
        sampling_ratio = self.config.get('sampling_ratio', 1.0)
        accuracies = []
        
        for model in cluster_models:
            self.model.load_state_dict(model.state_dict())
            acc = evaluate_erm(
                self.model, 
                dataloader=self.trainloader, 
                sampling_ratio=sampling_ratio,
                device=self.device
            )
            accuracies.append(acc)
            
        best_cluster = int(np.argmax(accuracies))
        return best_cluster

    def train_and_encode(self, best_cluster, cluster_models, orthogonal_keys):
        """
        1. Executes local training (backward pass) on selected cluster model ONLY.
        2. Applies VOMCA segment-wise over the full model parameter/gradient vector.
        """
        # Load chosen cluster model and run local training
        self.model.load_state_dict(cluster_models[best_cluster].state_dict())
        update = self.train()
        
        grad_vec = update['weights']
        if isinstance(grad_vec, torch.Tensor):
            grad_vec = grad_vec.detach().cpu().numpy()

        if isinstance(grad_vec, dict):
            flat_params = []
            for k, val in grad_vec.items():
                if isinstance(val, torch.Tensor):
                    flat_params.append(val.detach().cpu().numpy().flatten())
                elif isinstance(val, np.ndarray):
                    flat_params.append(val.flatten())
                else:
                    flat_params.append(np.array(val).flatten())
            grad_vec = np.concatenate(flat_params)
        elif isinstance(grad_vec, torch.Tensor):
            grad_vec = grad_vec.detach().cpu().numpy().flatten()
        elif isinstance(grad_vec, np.ndarray):
            grad_vec = grad_vec.flatten()
        else:
            grad_vec = np.array(grad_vec).flatten()
        
        n_clusters = self.config['n_clusters']
        M_i = orthogonal_keys[self.client_id]  # Shape: (dim, dim)
        dim = M_i.shape[0]                      # Block segment size l (e.g., 512)
        
        # One-hot cluster indicator vector C(i) padded to segment size dim
        c_i = np.zeros(dim)
        c_i[:n_clusters] = 0.0
        c_i[best_cluster] = 1.0
        
        # ----------------------------------------------------------------------
        # Option A: Segment-Wise VOMCA Matrix Encoding
        # ----------------------------------------------------------------------
        encoded_segments = []
        num_params = len(grad_vec)
        
        for start_idx in range(0, num_params, dim):
            chunk = grad_vec[start_idx : start_idx + dim]
            pad_len = dim - len(chunk)
            
            # Pad the final segment if it's smaller than `dim`
            if pad_len > 0:
                chunk = np.pad(chunk, (0, pad_len), mode='constant')
            
            # Add subtle zero-sum noise per segment
            noise_mask = np.random.normal(0, 0.001, size=M_i.shape)
            
            # Encrypt segment: chunk * M_i + C(i) * M_i[:n_clusters] + noise
            encoded_chunk = (
                np.dot(chunk, M_i) 
                + np.dot(c_i[:n_clusters], M_i[:n_clusters, :]) 
                + np.dot(np.ones(dim), noise_mask)
            )
            encoded_segments.append(encoded_chunk)
            
        # Concatenate encoded segments into a single flattened payload delta_i
        delta_i = np.concatenate(encoded_segments)
        
        return {
            'client_id': self.client_id,
            'best_cluster': best_cluster,
            'delta_i': delta_i,
            'original_len': num_params,
            'loss': update.get('loss', 0.0)
        }