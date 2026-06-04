from concurrent.futures import ThreadPoolExecutor
from fedlerate.servers.server_cfl import CFLServer
from fedlerate.clients.client_fedmix import FedMixClient
from fedlerate.models.models import load_model
import numpy as np


class FedMixServer(CFLServer):
    def __init__(self, config):
        super(FedMixServer, self).__init__(config)

    def init_clients(self):
        self.clients = self._init_clients_parallel(FedMixClient)
        
    def broadcast_model(self, clients, models):
        weights = [model.state_dict() for model in models]
        self._broadcast_parallel(clients, weights)

    def train(self):
        # Main training loop
        self.log.info("Training started")
        
        for rnd in range(self.config['n_rounds']):
            self.log.info(f"### Round {rnd} ###")

            selected_clients = self.select_clients()

            # Broadcast updated models to selected clients only for training
            self.broadcast_model(selected_clients, self.models)

            # Local training for selected clients
            n_jobs = self.config.get('n_jobs', 1)
            if n_jobs > 1:
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    updates = list(executor.map(lambda c: c.train(), selected_clients))
            else:
                updates = []
                for client in selected_clients:
                    update = client.train()
                    updates.append(update)

            # Aggregate updates
            self.aggregate(updates)

            total_loss = sum(update['loss'] for update in updates)

            # Evaluate - To save memory, we broadcast and evaluate one by one
            # The broadcast_model(self.clients, ...) call stores all models in all clients.
            # Instead, we'll rely on FedMixClient.evaluate() which now handles device movement.
            if (rnd + 1) % self.config.get('eval_every', 1) == 0 or (rnd + 1) == self.config['n_rounds']:
                # First, update all clients with the latest global mixture models
                weights = [model.state_dict() for model in self.models]
                self._broadcast_parallel(self.clients, weights)
                
                accs = self.evaluate()
                accuracy = sum(accs) / len(accs)
            else:
                accuracy = None

            metrics = {
                "round": rnd,
                "strategy": self.strategy,
                "model_name": self.config['model_name'],
                "accuracy": accuracy,
                "loss": total_loss / len(selected_clients)
            }
            self._save_metrics(metrics, self.config['metrics_save_path'])
            self._log_metrics(metrics)

    def aggregate(self, client_updates):
        if not client_updates:
            return
        total_data_size = sum(update['data_size'] for update in client_updates)
        if total_data_size == 0:
            return

        for k in range(self.config['n_clusters']):
            model_keys = client_updates[0]['weights'][k].keys()
            cum_weights = {}
            for key in model_keys:
                cum_weights[key] = sum(update['weights'][k][key] * update['data_size'] for update in client_updates) / total_data_size
            self.models[k].load_state_dict(cum_weights)