from fedlerate.servers.server_cfl import CFLServer
from fedlerate.clients.client_base import BaseClient


class FCFLServer(CFLServer):
    def __init__(self, config):
        super(FCFLServer, self).__init__(config)

    def train(self):
        # Main training loop
        self.log.info("Training started")

        cluster_labels = self.config['labels']

        for rnd in range(self.config['n_rounds']):
            self.log.info(f"### Round {rnd} ###")

            selected_clients = self.select_clients()
            cluster_assignments = [[] for _ in range(self.config['n_clusters'])]
            for client in selected_clients:
                client_id = client.client_id
                label = cluster_labels[client_id]
                cluster_assignments[label].append(client_id)

            # Train each cluster model with its assigned clients
            total_loss = 0
            for cid in range(self.config['n_clusters']):
                # Local training
                updates = []
                self.broadcast_model([self.clients[client_id] for client_id in cluster_assignments[cid]], self.models[cid])
                for client_id in cluster_assignments[cid]:
                    update = self.clients[client_id].train()
                    updates.append(update)

                # Aggregate updates
                if len(cluster_assignments[cid]) > 0:
                    self.aggregate(updates, cid)

                total_loss += sum(update['loss'] for update in updates)

            # Evaluate cluster models
            for cid, cluster in enumerate(cluster_assignments):
                self.broadcast_model([self.clients[client_id] for client_id in cluster], self.models[cid])

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

    def init_clients(self):
        self.clients = [BaseClient(client_id=i, config=self.config, device=self.device) for i in range(self.config['n_clients'])]