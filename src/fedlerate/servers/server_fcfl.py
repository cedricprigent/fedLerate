from concurrent.futures import ThreadPoolExecutor
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
            n_jobs = self.config.get('n_jobs', 1)
            for cid in range(self.config['n_clusters']):
                # Local training
                self.broadcast_model([self.clients[client_id] for client_id in cluster_assignments[cid]], self.models[cid])
                
                if n_jobs > 1:
                    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                        updates = list(executor.map(lambda client_id: self.clients[client_id].train(), cluster_assignments[cid]))
                else:
                    updates = []
                    for client_id in cluster_assignments[cid]:
                        update = self.clients[client_id].train()
                        updates.append(update)

                # Aggregate updates
                if len(cluster_assignments[cid]) > 0:
                    self.aggregate(updates, cid)

                total_loss += sum(update['loss'] for update in updates)

            # Evaluate cluster models
            if (rnd + 1) % self.config.get('eval_every', 1) == 0 or (rnd + 1) == self.config['n_rounds']:
                for cid, cluster in enumerate(cluster_assignments):
                    self.broadcast_model([self.clients[client_id] for client_id in cluster], self.models[cid])

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

    def init_clients(self):
        self.clients = self._init_clients_parallel(BaseClient)