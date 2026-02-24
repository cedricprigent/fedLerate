from fedlerate.servers.server_cfl import CFLServer
from fedlerate.clients.client_fedmix import FedMixClient
from fedlerate.models.models import load_model
import numpy as np


class FedMixServer(CFLServer):
    def __init__(self, config):
        super(FedMixServer, self).__init__(config)

    def init_clients(self):
        self.clients = [FedMixClient(client_id=i, config=self.config, device=self.device) for i in range(self.config['n_clients'])]
        
    def broadcast_model(self, clients, models):
        for client in clients:
            client.set_model_weights([model.state_dict() for model in models])

    def train(self):
        # Main training loop
        self.log.info("Training started")
        
        for rnd in range(self.config['n_rounds']):
            self.log.info(f"### Round {rnd} ###")

            selected_clients = self.select_clients()

            # Train each cluster model with its assigned clients
            updates = []
            for client in self.clients:
                # Local training
                self.broadcast_model(selected_clients, self.models)

                update = client.train()
                updates.append(update)

            # Aggregate updates
            self.aggregate(updates)

            total_loss = sum(update['loss'] for update in updates)

            # Evaluate
            self.broadcast_model(selected_clients, self.models)
            accs = self.evaluate()

            metrics = {
                "round": rnd,
                "strategy": self.strategy,
                "model_name": self.config['model_name'],
                "accuracy": sum(accs) / len(accs),
                "loss": total_loss / len(selected_clients)
            }
            self._save_metrics(metrics, self.config['metrics_save_path'])
            self._log_metrics(metrics)

    def aggregate(self, client_updates):
        total_data_size = sum(update['data_size'] for update in client_updates)
        for k in range(self.config['n_clusters']):
            cum_weights = {}
            for key in self.models[k].state_dict().keys():
                cum_weights[key] = sum(update['weights'][k][key] * update['data_size'] for update in client_updates) / total_data_size
            self.models[k].load_state_dict(cum_weights)