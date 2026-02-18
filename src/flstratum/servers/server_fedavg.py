from flstratum.servers.server_abc import ServerABC
from flstratum.clients.client_base import BaseClient
from flstratum.models.models import load_model
import numpy as np
import logging

class FedAvgServer(ServerABC):
    def __init__(self, config):
        super(FedAvgServer, self).__init__(config)

    def train(self):
        # Main training loop
        self.log.info("Training started")
        self.broadcast_model()
        
        for rnd in range(self.config['n_rounds']):
            self.log.info(f"### Round {rnd} ###")

            selected_clients = self.select_clients()

            client_updates = []
            for client in selected_clients:
                client_updates.append(client.train())

            self.aggregate(client_updates)
            self.broadcast_model()

            accs = self.evaluate()

            metrics = {
                "round": rnd,
                "strategy": self.strategy,
                "model_name": self.config['model_name'],
                "accuracy": sum(accs) / len(accs),
                "loss": sum(update['loss'] for update in client_updates) / len(client_updates)
            }
            self._save_metrics(metrics, self.config['metrics_save_path'])
            self._log_metrics(metrics)

    def aggregate(self, client_updates):
        # FedAvg aggregation logic
        total_data_size = sum(update['data_size'] for update in client_updates)
        cum_weights = {}
        for key in self.global_model.state_dict().keys():
            cum_weights[key] = sum(update['weights'][key] * update['data_size'] for update in client_updates) / total_data_size
        self.global_model.load_state_dict(cum_weights)

    def evaluate(self):
        # Evaluation global model on client data
        accs = []
        for client in self.clients:
            acc = client.evaluate()
            accs.append(acc)
        return accs

    def init_clients(self):
        # Initialize clients based on the configuration
        self.clients = [BaseClient(client_id=i, config=self.config, device=self.device) for i in range(self.config['n_clients'])]

    def select_clients(self):
        # Client selection logic (e.g., random selection)
        n_clients = len(self.clients)
        if n_clients < self.config['n_clients_per_round']:
            return self.clients
        indices = np.random.choice(n_clients, self.config['n_clients_per_round'], replace=False)
        return [self.clients[i] for i in indices]

    def init_model(self) -> None:
        self.global_model = load_model(self.config['model_name'], device='cpu')

    def broadcast_model(self):
        # Broadcast the global model to all clients
        for client in self.clients:
            client.set_model_weights(self.global_model.state_dict())

    def run(self):
        self.train()