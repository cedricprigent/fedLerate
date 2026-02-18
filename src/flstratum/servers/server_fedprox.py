from flstratum.servers.server_fedavg import FedAvgServer
from flstratum.models.functions import train_fedprox, evaluate
from flstratum.clients.client_fedprox import FedProxClient


class FedProxServer(FedAvgServer):
    def __init__(self, config):
        super(FedProxServer, self).__init__(config)

    def init_clients(self):
        # Initialize clients based on the configuration
        self.clients = [FedProxClient(client_id=i, config=self.config, device=self.device) for i in range(self.config['n_clients'])]