from fedlerate.servers.server_cfl import CFLServer
from fedlerate.clients.client_base import BaseClient
from fedlerate.models.models import load_model
import numpy as np


class IFCAServer(CFLServer):
    def __init__(self, config):
        super(IFCAServer, self).__init__(config)

    def init_clients(self):
        self.clients = [BaseClient(client_id=i, config=self.config, device=self.device) for i in range(self.config['n_clients'])]

    def assign_clusters(self, selected_clients):
        cluster_assignments = [[] for _ in range(self.config['n_clusters'])]
        for client in selected_clients:
            losses = []
            for cluster in range(self.config['n_clusters']):
                self.broadcast_model([client], self.models[cluster])
                loss = client.evaluate()
                losses.append(loss)
            best_cluster = np.argmin(losses)
            cluster_assignments[best_cluster].append(client.client_id)

        return cluster_assignments

    def train(self):
        # Main training loop
        self.log.info("Training started")
        
        for rnd in range(self.config['n_rounds']):
            self.log.info(f"### Round {rnd} ###")

            selected_clients = self.select_clients()

            # Assign clients to clusters
            if rnd == 0:
                # Random assignment
                cids = np.arange(0, len(self.clients))
                shuffle = np.random.shuffle(cids)
                clusters = np.array_split(cids, self.config['n_clusters'])
                cluster_assignments = [c.tolist() for c in clusters]
            else:
                # ERM assignment
                cluster_assignments = self.assign_clusters(selected_clients)

            # Train each cluster model with its assigned clients
            total_loss = 0
            for cid, cluster in enumerate(cluster_assignments):
                # Local training
                updates = []
                self.broadcast_model([self.clients[client_id] for client_id in cluster], self.models[cid])
                for client_id in cluster:
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