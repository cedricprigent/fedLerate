import gc
from fedlerate.clients.client_fedmix import FedMixClient
from fedlerate.servers.server_cfl import CFLServer


class FedMixServer(CFLServer):
    """Memory-optimized Server for FedMix.

    Manages global mixture models and orchestrates federated training and
    evaluation rounds without overloading RAM/GPU VRAM.
    """

    def __init__(self, config):
        super(FedMixServer, self).__init__(config)

    def init_clients(self):
        """Initialize all clients using the base server initialization."""
        self.clients = self._init_clients_parallel(FedMixClient)

    def train(self):
        """Main Federated Learning Training Loop."""
        self.log.info("Training started")

        for rnd in range(self.config['n_rounds']):
            self.log.info(f"### Round {rnd} ###")

            # 1. Select clients for the current round
            selected_clients = self.select_clients()

            # 2. Extract global state dicts ONCE on CPU to share across workers
            global_weights = [model.state_dict() for model in self.models]

            # 3. Perform local client training sequentially to prevent VRAM spikes
            updates = []
            for client in selected_clients:
                # Pass global weights directly to stateless client.train()
                update = client.train(global_weights=global_weights)
                updates.append(update)

            # 4. Aggregate updates into global models
            self.aggregate(updates)

            # Calculate average training loss for selected clients
            total_loss = sum(update['loss'] for update in updates)
            avg_loss = total_loss / len(selected_clients) if selected_clients else 0.0

            # 5. Free client update dictionaries from RAM
            del updates
            del global_weights
            gc.collect()

            # 6. Periodic Evaluation Phase
            eval_every = self.config.get('eval_every', 1)
            is_last_round = (rnd + 1) == self.config['n_rounds']

            if (rnd + 1) % eval_every == 0 or is_last_round:
                self.log.info(f"Evaluating round {rnd}...")
                
                # Fetch fresh aggregated global weights
                eval_weights = [model.state_dict() for model in self.models]
                accs = []
                
                # Evaluate on all clients sequentially
                for client in self.clients:
                    acc = client.evaluate(global_weights=eval_weights)
                    accs.append(acc)

                accuracy = sum(accs) / len(accs) if accs else 0.0
                
                del eval_weights
                gc.collect()
            else:
                accuracy = None

            # 7. Metrics Logging & Saving
            metrics = {
                "round": rnd,
                "strategy": self.strategy,
                "model_name": self.config['model_name'],
                "accuracy": accuracy,
                "loss": avg_loss
            }
            self._save_metrics(metrics, self.config['metrics_save_path'])
            self._log_metrics(metrics)

    def aggregate(self, client_updates):
        """Weighted average aggregation for Mixture of Experts (K clusters)."""
        if not client_updates:
            return

        total_data_size = sum(update['data_size'] for update in client_updates)
        if total_data_size == 0:
            return

        n_clusters = self.config['n_clusters']

        # Aggregate weights for each mixture component k independently
        for k in range(n_clusters):
            model_keys = client_updates[0]['weights'][k].keys()
            cum_weights = {}

            for key in model_keys:
                # Compute weighted sum across all client updates
                weighted_sum = sum(
                    update['weights'][k][key] * update['data_size']
                    for update in client_updates
                )
                cum_weights[key] = weighted_sum / float(total_data_size)

            # Load newly aggregated weights into the server's global model
            self.models[k].load_state_dict(cum_weights)