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
        
        def get_best_cluster(client):
            losses = []
            for cluster in range(self.config['n_clusters']):
                self.broadcast_model([client], self.models[cluster])
                loss = client.evaluate()
                losses.append(loss)
            return client.client_id, np.argmin(losses)

        results = [get_best_cluster(c) for c in selected_clients]

        for client_id, best_cluster in results:
            cluster_assignments[best_cluster].append(client_id)

        return cluster_assignments

    def train(self):
        # Main training loop
        self.log.info("Training started")
        
        for rnd in range(self.config['n_rounds']):
            self.log.info(f"### Round {rnd} ###")

            selected_clients = self.select_clients()

            # Warmup phase - Run FedAvg for a few rounds to get initial models
            if rnd < 5:
                # Assign all clients to a single cluster and train with FedAvg
                cluster_assignments = [[] for _ in range(self.config['n_clusters'])]
                cluster_assignments[0] = [client.client_id for client in selected_clients]
            elif rnd == 5:
                # Copy global model to all cluster models
                for cid in range(self.config['n_clusters']):
                    self.models[cid].load_state_dict(self.models[0].state_dict())

                # Random assignment
                cids = np.arange(0, len(self.clients))
                np.random.shuffle(cids)
                clusters = np.array_split(cids, self.config['n_clusters'])
                cluster_assignments = [c.tolist() for c in clusters]
            else:
                # ERM assignment
                cluster_assignments = self.assign_clusters(selected_clients)

            # Train each cluster model with its assigned clients
            total_loss = 0
            n_jobs = self.config.get('n_jobs', 1)
            for cid, cluster in enumerate(cluster_assignments):
                # Local training
                self.broadcast_model([self.clients[client_id] for client_id in cluster], self.models[cid])
                
                updates = []
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