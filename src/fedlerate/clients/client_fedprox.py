from fedlerate.clients.client_base import BaseClient
from fedlerate.models.functions import train_fedprox, evaluate


class FedProxClient(BaseClient):
    def __init__(self, client_id, config, device):
        super(FedProxClient, self).__init__(client_id, config, device)

    def train(self):
        # Train local model on client data
        # Return model update and data size for aggregation
        loss, _ = train_fedprox(self.model, 
            dataloader=self.trainloader, 
            epochs=self.config['local_epochs'], 
            device=self.device,
            mu=self.config['fedprox_mu'],
            lr=self.config['local_lr'],
            noise_mixer=self.noise_mixer
        )

        self.config['local_lr'] *= self.config.get('lr_decay', 1.0)

        return {
            "weights": self.get_model_weights(),
            "data_size": self.trainloader.dataset.__len__(),
            "loss": loss
        }