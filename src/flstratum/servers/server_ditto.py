from flstratum.servers.server_fedavg import FedAvgServer
from flstratum.clients.client_ditto import DittoClient

class DittoServer(FedAvgServer):
    def __init__(self, config):
        super(DittoServer, self).__init__(config)

    def init_clients(self):
        # Initialize clients based on the configuration
        self.clients = [DittoClient(client_id=i, config=self.config, device=self.device) for i in range(self.config['n_clients'])]