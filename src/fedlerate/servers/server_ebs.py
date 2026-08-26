import numpy as np
from concurrent.futures import ThreadPoolExecutor
from fedlerate.servers.server_cfl import CFLServer
from fedlerate.clients.client_ebs import ClientEBS


class EBSCFLServer(CFLServer):
    def __init__(self, config):
        super(EBSCFLServer, self).__init__(config)
        self.strategy = "EBS-CFL (Segment-Wise VOMCA)"
        self._init_orthogonal_keys()

    def _init_orthogonal_keys(self):
        """Generates random orthogonal matrices M_i for VOMCA secret key distribution."""
        n_clients = len(self.config['client_ids']) if 'client_ids' in self.config else 100
        dim = self.config.get('embedding_dim', 512)
        
        self.orthogonal_keys = {}
        for cid in range(n_clients):
            H = np.random.randn(dim, dim)
            Q, _ = np.linalg.qr(H)
            self.orthogonal_keys[cid] = Q

    def init_clients(self):
        self.clients = self._init_clients_parallel(ClientEBS)

    def train(self):
        self.log.info("EBS-CFL (Segment-Wise VOMCA) Training Started")
        n_clusters = self.config['n_clusters']
        n_jobs = self.config.get('n_jobs', 1)

        for rnd in range(self.config['n_rounds']):
            self.log.info(f"### Round {rnd} ###")
            selected_clients = self.select_clients()

            # Initial Warmup Phase (FedAvg over single global model for first 3 rounds)
            if rnd < 3:
                cluster_assignments = [[] for _ in range(n_clusters)]
                cluster_assignments[0] = [client.client_id for client in selected_clients]
                
                for cid, cluster in enumerate(cluster_assignments):
                    if len(cluster) == 0:
                        continue
                    self.broadcast_model([self.clients[c] for c in cluster], self.models[cid])
                    if n_jobs > 1:
                        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                            updates = list(executor.map(lambda c: self.clients[c].train(), cluster))
                    else:
                        updates = [self.clients[c].train() for c in cluster]
                    self.aggregate(updates, cid)
                continue
            elif rnd == 3:
                # Copy global base model parameters to all cluster models
                for cid in range(n_clusters):
                    self.models[cid].load_state_dict(self.models[0].state_dict())

            # -------------------------------------------------------------
            # Step 1: Client Cluster Selection & Segment-Wise VOMCA Encoding
            # -------------------------------------------------------------
            def client_step(client):
                best_cluster = client.select_best_cluster(self.models)
                return client.train_and_encode(
                    best_cluster=best_cluster,
                    cluster_models=self.models,
                    orthogonal_keys=self.orthogonal_keys
                )

            if n_jobs > 1:
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    client_payloads = list(executor.map(client_step, selected_clients))
            else:
                client_payloads = [client_step(c) for c in selected_clients]

            # -------------------------------------------------------------
            # Step 2: Segment-Wise VOMCA Decoding & Aggregation
            # -------------------------------------------------------------
            cluster_assignments = [[] for _ in range(n_clusters)]
            cluster_updates = [[] for _ in range(n_clusters)]
            total_loss = 0.0

            for payload in client_payloads:
                cid = payload['best_cluster']
                client_id = payload['client_id']
                delta_i = payload['delta_i']
                orig_len = payload['original_len']
                
                M_i = self.orthogonal_keys[client_id]
                dim = M_i.shape[0]
                
                # Reconstruct original vector segment by segment
                decoded_segments = []
                for start_idx in range(0, len(delta_i), dim):
                    encoded_chunk = delta_i[start_idx : start_idx + dim]
                    # Multiply by transpose of orthogonal matrix (M_i^T)
                    decoded_chunk = np.dot(encoded_chunk, M_i.T)
                    decoded_segments.append(decoded_chunk)
                    
                # Truncate any zero-padding applied during client encoding
                reconstructed_update = np.concatenate(decoded_segments)[:orig_len]
                
                cluster_assignments[cid].append(client_id)
                cluster_updates[cid].append({
                    'model_params': reconstructed_update,
                    'weight': 1.0
                })
                total_loss += payload['loss']

            # -------------------------------------------------------------
            # Step 3: Cluster Parameter Aggregation
            # -------------------------------------------------------------
            for cid in range(n_clusters):
                if len(cluster_updates[cid]) > 0:
                    self.aggregate(cluster_updates[cid], cid)

            # Metrics Evaluation & Logging
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
                "loss": total_loss / len(selected_clients)
            }
            self._save_metrics(metrics, self.config['metrics_save_path'])
            self._log_metrics(metrics)