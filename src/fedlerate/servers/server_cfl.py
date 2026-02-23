from abc import ABC, abstractmethod
from fedlerate.servers.server_abc import ServerABC
from fedlerate.models.models import load_model
import numpy as np


class CFLServer(ServerABC):
    def __init__(self, config):
        super(CFLServer, self).__init__(config)

    def init_model(self) -> None:
        self.models = [load_model(self.config['model_name'], device='cpu') for _ in range(self.config['n_clusters'])]

    def broadcast_model(self, clients, model):
        for client in clients:
            client.set_model_weights(model.state_dict())
    
    def select_clients(self):
        n_clients = len(self.clients)
        if n_clients < self.config['n_clients_per_round']:
            return self.clients
        indices = np.random.choice(n_clients, self.config['n_clients_per_round'], replace=False)
        return [self.clients[i] for i in indices]

    def aggregate(self, client_updates, cluster_id):
        total_data_size = sum(update['data_size'] for update in client_updates)
        cum_weights = {}
        for key in self.models[cluster_id].state_dict().keys():
            cum_weights[key] = sum(update['weights'][key] * update['data_size'] for update in client_updates) / total_data_size
        self.models[cluster_id].load_state_dict(cum_weights)

    def evaluate(self):
        # Evaluate the global model on the validation set
        self.log.info("Evaluation started")
        accs = []
        for client in self.clients:
            acc = client.evaluate()
            accs.append(acc)
        return accs

    @abstractmethod
    def train(self):
        pass    

    @abstractmethod
    def init_clients(self):
        pass

    def run(self):
        self.train()