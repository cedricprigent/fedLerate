from fedlerate.clients.client_abc import ClientABC
from fedlerate.models.functions import train_fedmix, evaluate_fedmix
from fedlerate.models.models import load_model
from fedlerate.data.partition import load_partition

import torch

class FedMixClient(ClientABC):
    def __init__(self, client_id, config, device):
        super(FedMixClient, self).__init__(client_id, config, device)
        self.init_model(self.config['n_clusters'])
        self.init_data()

    def train(self):
        # Move models to device
        for model in self.models:
            model.to(self.device)

        # Train local model on client data
        # Return model update and data size for aggregation
        loss, _ = train_fedmix(self.models,
            self.weights, 
            dataloader=self.trainloader, 
            epochs=self.config['local_epochs'], 
            lr=self.config['local_lr'],
            device=self.device,
            noise_mixer=self.noise_mixer
        )

        self.config['local_lr'] *= self.config.get('lr_decay', 1.0)

        weights = self.get_model_weights()

        # Move models back to CPU
        for model in self.models:
            model.to('cpu')
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return {
            "weights": weights,
            "data_size": self.trainloader.dataset.__len__(),
            "loss": loss
        }

    def evaluate(self):
        # Move models to device
        for model in self.models:
            model.to(self.device)

        # Evaluate local model on client test data
        # Return accuracy or other metrics
        accuracy = evaluate_fedmix(self.models,
            self.weights, 
            dataloader=self.testloader, 
            device=self.device
        )

        # Move models back to CPU
        for model in self.models:
            model.to('cpu')
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return accuracy

    def init_model(self, k):
        # Initialize mixture of experts
        self.weights = torch.nn.Parameter(torch.ones(k, device=self.device) / k)

        # Initialize models on CPU to save GPU memory. They will be moved to device during train/evaluate.
        self.models = [
            load_model(self.config['model_name'], device='cpu') for _ in range(k)
        ]

    def init_data(self):
        # Initialize local data for the client
        self.trainloader, self.testloader, _ = load_partition(
            num_partition=self.client_id, 
            batch_size=self.config['batch_size']
        )

    def set_model_weights(self, weights):
        if self.models is not None:
            for i, w in enumerate(weights):
                self.models[i].load_state_dict(w)
        else:
            return None

    def get_model_weights(self):
        if self.models is not None:
            return [model.state_dict() for model in self.models]
        else:
            return None