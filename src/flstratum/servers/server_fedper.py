from flstratum.servers.server_fedavg import FedAvgServer
from flstratum.models.models import load_model
from flstratum.clients.client_pfl import PFLClient 


class FedPerServer(FedAvgServer):
    def __init__(self, config):
        super(FedPerServer, self).__init__(config)

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

    def init_model(self) -> None:
        self.global_model = load_model(self.config['model_name'], device='cpu').global_layers

    def init_clients(self):
        # Initialize clients based on the configuration
        self.clients = [PFLClient(client_id=i, config=self.config, device=self.device) for i in range(self.config['n_clients'])]