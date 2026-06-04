from concurrent.futures import ThreadPoolExecutor
from fedlerate.servers.server_fedavg import FedAvgServer
from fedlerate.models.models import load_model
from fedlerate.clients.client_pfl import PFLClient 


class FedPerServer(FedAvgServer):
    def __init__(self, config):
        super(FedPerServer, self).__init__(config)

    def train(self):
        # Main training loop
        self.log.info("Training started")

        warmup_rounds = 20
        is_warmup = warmup_rounds > 0
        for client in self.clients:
            client.return_full_model = is_warmup

        self.broadcast_model()

        for rnd in range(self.config['n_rounds']):
            if rnd == warmup_rounds:
                self.log.info(f"Warmup finished. Switching to FedPer (aggregating only global layers).")
                self.global_model = self.full_model.global_layers
                for client in self.clients:
                    client.return_full_model = False

            self.log.info(f"### Round {rnd} ###")

            selected_clients = self.select_clients(rnd)

            n_jobs = self.config.get('n_jobs', 1)
            if n_jobs > 1:
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    client_updates = list(executor.map(lambda c: c.train(), selected_clients))
            else:
                client_updates = []
                for client in selected_clients:
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

    def init_model(self) -> None:
        self.full_model = load_model(self.config['model_name'], device='cpu')
        self.global_model = self.full_model

    def init_clients(self):
        # Initialize clients based on the configuration
        self.clients = self._init_clients_parallel(PFLClient)