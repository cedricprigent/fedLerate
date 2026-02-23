from flstratum.clients.client_abc import ClientABC
from flstratum.models.functions import train_fedmix, evaluate_fedmix
from flstratum.models.models import load_model
from flstratum.data.partition import load_partition

import torch

class FedMixClient(ClientABC):
    def __init__(self, client_id, config, device):
        super(FedMixClient, self).__init__(client_id, config, device)
        self.init_model(self.config['n_clusters'])
        self.init_data()

    def train(self):
        # Train local model on client data
        # Return model update and data size for aggregation
        loss, _ = train_fedmix(self.models,
            self.weights, 
            dataloader=self.trainloader, 
            epochs=self.config['local_epochs'], 
            lr=self.config['local_lr'],
            device=self.device
        )

        return {
            "weights": self.get_model_weights(),
            "data_size": self.trainloader.dataset.__len__(),
            "loss": loss
        }

    def evaluate(self):
        # Evaluate local model on client test data
        # Return accuracy or other metrics
        accuracy = evaluate_fedmix(self.models,
            self.weights, 
            dataloader=self.testloader, 
            device=self.device
        )
        return accuracy

    def init_model(self, k):
        # Initialize mixture of experts
        self.weights = torch.nn.Parameter(torch.ones(k)).to(self.device) / k
        self.models = [
            load_model(self.config['model_name'], device=self.device) for _ in range(k)
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
            return [model[i].state_dict() for model in self.models]
        else:
            return None