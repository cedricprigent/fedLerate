from concurrent.futures import ThreadPoolExecutor
from fedlerate.servers.server_abc import ServerABC
from fedlerate.clients.client_base import BaseClient
from fedlerate.models.models import load_model
import numpy as np
import logging

class FedAvgServer(ServerABC):
    def __init__(self, config):
        super(FedAvgServer, self).__init__(config)

    def train(self):
        # Main training loop
        self.log.info(f"Training started for {self.config['n_rounds']} rounds with strategy: {self.strategy}")
        self.broadcast_model()
        
        for rnd in range(self.config['n_rounds']):
            self.log.info(f"### Round {rnd} ###")

            self.sampled_clients = self.select_clients(rnd)

            n_jobs = self.config.get('n_jobs', 1)
            if n_jobs > 1:
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    client_updates = list(executor.map(lambda c: c.train(), self.sampled_clients))
            else:
                client_updates = []
                for client in self.sampled_clients:
                    client_updates.append(client.train())

            self.aggregate(client_updates)
            self.broadcast_model()

            if (rnd + 1) % self.config.get('eval_every', 1) == 0 or (rnd + 1) == self.config['n_rounds']:
                accs = self.evaluate()
                accuracy = sum(accs) / len(accs)
            else:
                accuracy = None

            metrics = {
                "round": rnd,
                "strategy": self.strategy,
                "model_name": self.config['model_name'],
                "accuracy": accuracy,
                "loss": sum(update['loss'] for update in client_updates) / len(client_updates)
            }
            self._save_metrics(metrics, self.config['metrics_save_path'])
            self._log_metrics(metrics)

        return self.global_model

    def aggregate(self, client_updates):
        # FedAvg aggregation logic
        total_data_size = sum(update['data_size'] for update in client_updates)
        cum_weights = {}
        for key in self.global_model.state_dict().keys():
            cum_weights[key] = sum(update['weights'][key] * update['data_size'] for update in client_updates) / total_data_size
        self.global_model.load_state_dict(cum_weights)

    def evaluate(self):
        # Evaluation global model on client data
        n_jobs = self.config.get('n_jobs', 1)
        if n_jobs > 1:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                accs = list(executor.map(lambda c: c.evaluate(), self.clients))
        else:
            accs = []
            for client in self.clients:
                accs.append(client.evaluate())
        return accs

    def init_clients(self):
        # Initialize clients based on the configuration
        self.clients = self._init_clients_parallel(BaseClient)

    def select_clients(self, rnd):
        # Client selection logic (e.g., random selection)
        n_clients = len(self.clients)
        if n_clients < self.config['n_clients_per_round']:
            return self.clients
        
        # Set seed based on global seed and round for reproducibility
        np.random.seed(self.config['seed'] + rnd)
        indices = np.random.choice(n_clients, self.config['n_clients_per_round'], replace=False)
        return [self.clients[i] for i in indices]

    def init_model(self) -> None:
        self.global_model = load_model(self.config['model_name'], n_classes=self.config["n_classes"], device='cpu')

    def broadcast_model(self):
        # Broadcast the global model to all clients
        self._broadcast_parallel(self.clients, self.global_model.state_dict())

    def run(self):
        self.train()