from fedlerate.clients.client_base import BaseClient
from fedlerate.models.functions import train, evaluate, train_ditto
from fedlerate.models.models import load_model


class DittoClient(BaseClient):
    def __init__(self, client_id, config, device):
        super(DittoClient, self).__init__(client_id, config, device)

    def train(self):
        # Train local model on client data
        # Return model update and data size for aggregation
        loss, _ = train(self.model, 
            dataloader=self.trainloader, 
            epochs=self.config['local_epochs'], 
            lr=self.config['local_lr'],
            device=self.device
        )

        loss, _ = train_ditto(
            personalized_model=self.personalized_model, 
            global_model=self.model, 
            dataloader=self.trainloader, 
            epochs=self.config['local_epochs'], 
            lr=self.config['local_lr'],
            lambda_=self.config['ditto_lambda'],
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
        accuracy = evaluate(self.personalized_model, 
            dataloader=self.testloader, 
            device=self.device
        )
        return accuracy

    def init_model(self):
        self.model = load_model(self.config['model_name'], device=self.device)
        self.personalized_model = load_model(self.config['model_name'], device=self.device)