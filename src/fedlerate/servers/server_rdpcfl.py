from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from opacus.accountants.utils import get_noise_multiplier

from fedlerate.servers.server_cfl import CFLServer
from fedlerate.clients.client_rdpcfl import RDPCFLClient


def extract_flat_params(model):
    """Flattens all trainable parameters into a single 1D tensor."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model, flat_params):
    """Updates a model's state using a 1D parameter vector."""
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[idx:idx + numel].view_as(p.data))
        idx += numel


class RDPCFLServer(CFLServer):
    def __init__(self, config):
        if config.get('epsilon') is not None:
            target_epsilon = config['epsilon']
            target_delta = config.get('delta') or 1e-5
            sample_rate = config['n_clients_per_round'] / config['n_clients']
            epochs = config['n_rounds']

            noise_multiplier = get_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                sample_rate=sample_rate,
                epochs=epochs
            )
            config['noise_multiplier'] = noise_multiplier
            import logging
            logging.getLogger(__name__).info(
                f"DP: epsilon={target_epsilon}, delta={target_delta} → "
                f"noise_multiplier={noise_multiplier:.4f} "
                f"(sample_rate={sample_rate:.4f}, rounds={epochs})"
            )

        super(RDPCFLServer, self).__init__(config)
        self.client_clusters = {}

    def init_clients(self):
        self.clients = self._init_clients_parallel(RDPCFLClient)

    def initial_gmm_clustering(self, client_updates_dict):
        """Stage 1: Perform GMM clustering on noisy DP updates."""
        client_ids = list(client_updates_dict.keys())
        updates_matrix = np.array([client_updates_dict[cid].numpy() for cid in client_ids])

        # Fit Gaussian Mixture Model to handle noise variance
        gmm = GaussianMixture(n_components=self.config['n_clusters'], covariance_type='diag', random_state=self.config.get('seed', 42))
        cluster_labels = gmm.fit_predict(updates_matrix)

        for cid, label in zip(client_ids, cluster_labels):
            self.client_clusters[cid] = int(label)

    def refine_clusters_by_loss(self, clients):
        """
        Stage 2: Loss-based Refinement.
        Evaluates each client on all cluster models to mitigate DP noise misalignment.
        """
        def evaluate_client_losses(client):
            losses = []
            for k in range(self.config['n_clusters']):
                loss_k = client.evaluate_loss(self.models[k].state_dict())
                losses.append(loss_k)
            # Reassign client to cluster yielding the minimum empirical loss
            best_cluster = int(np.argmin(losses))
            return client.client_id, best_cluster

        n_jobs = self.config.get('n_jobs', 1)
        if n_jobs > 1:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(evaluate_client_losses, clients))
        else:
            results = [evaluate_client_losses(client) for client in clients]

        for client_id, best_cluster in results:
            self.client_clusters[client_id] = best_cluster

    def aggregate_cluster_models(self, client_updates_dict):
        """Aggregate DP updates within each assigned cluster."""
        for k in range(self.config['n_clusters']):
            # Collect updates for cluster k
            cluster_cids = [cid for cid, cluster_idx in self.client_clusters.items() if cluster_idx == k and cid in client_updates_dict]
            if not cluster_cids:
                continue

            # Average updates
            avg_update = torch.stack([client_updates_dict[cid] for cid in cluster_cids]).mean(dim=0)

            # Apply aggregated update to Cluster Model k
            current_params = extract_flat_params(self.models[k])
            new_params = current_params + avg_update.to(current_params.device)
            set_flat_params(self.models[k], new_params)

    def train(self):
        self.log.info("Training started")

        for rnd in range(self.config['n_rounds']):
            self.log.info(f"### Round {rnd} ###")

            selected_clients = self.select_clients()

            # Step 1: Perform Clustering / Refinement for subsequent rounds before training
            if rnd > 0:
                self.refine_clusters_by_loss(selected_clients)
                self.log.info(f"Round {rnd+1} Loss-Refined Clusters: {self.client_clusters}")

            # Broadcast the currently assigned cluster model to each client
            for client in selected_clients:
                # Default to cluster 0 if not assigned yet
                current_cluster = self.client_clusters.get(client.client_id, 0)
                self.broadcast_model([client], self.models[current_cluster])

            # Train clients (can be parallelized using ThreadPoolExecutor)
            n_jobs = self.config.get('n_jobs', 1)
            if n_jobs > 1:
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    results = list(executor.map(lambda c: (c.client_id, c.train()), selected_clients))
            else:
                results = [(c.client_id, c.train()) for c in selected_clients]

            client_dp_updates = {}
            client_updates = []
            for client_id, update in results:
                client_dp_updates[client_id] = update['dp_update']
                client_updates.append(update)

            # Step 2: Perform Clustering for the initial round
            if rnd == 0:
                # Initial Round: GMM-based assignment from updates
                self.initial_gmm_clustering(client_dp_updates)
                self.log.info(f"Round {rnd+1} Initial GMM Clusters: {self.client_clusters}")

            # Step 3: Aggregate DP Updates Per Cluster
            self.aggregate_cluster_models(client_dp_updates)

            total_loss = sum(update['loss'] for update in client_updates)

            # Evaluate cluster models
            if (rnd + 1) % self.config.get('eval_every', 1) == 0 or (rnd + 1) == self.config['n_rounds']:
                # Broadcast models to their assigned clients for evaluation
                for cid in range(self.config['n_clusters']):
                    cluster_clients = [c for c in self.clients if self.client_clusters.get(c.client_id, 0) == cid]
                    if cluster_clients:
                        self.broadcast_model(cluster_clients, self.models[cid])

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
