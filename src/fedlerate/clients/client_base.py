from fedlerate.clients.client_abc import ClientABC
from fedlerate.models.functions import train, evaluate
from fedlerate.models.models import load_model
from fedlerate.data.partition import load_partition


class BaseClient(ClientABC):
    def __init__(self, client_id, config, device):
        super(BaseClient, self).__init__(client_id, config, device)
        self.init_model()
        self.init_data()

    def train(self):
        # Train local model on client data
        # Return model update and data size for aggregation
        loss, _ = train(self.model, 
            dataloader=self.trainloader, 
            epochs=self.config['local_epochs'], 
            lr=self.config['local_lr'],
            device=self.device,
            noise_mixer=self.noise_mixer
        )

        self.config['local_lr'] *= self.config.get('lr_decay', 1.0)

        return {
            "weights": self.get_model_weights(),
            "data_size": self.trainloader.dataset.__len__(),
            "loss": loss
        }

    def evaluate(self):
        # Evaluate local model on client test data
        # Return accuracy or other metrics
        accuracy = evaluate(self.model, 
            dataloader=self.testloader, 
            device=self.device
        )
        return accuracy

    def init_model(self):
        self.model = load_model(self.config['model_name'], device=self.device)

    def init_data(self):
        # Initialize local data for the client
        self.trainloader, self.testloader, _ = load_partition(
            num_partition=self.client_id, 
            batch_size=self.config['batch_size']
        )