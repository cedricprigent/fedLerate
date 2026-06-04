from fedlerate.servers.server_fedavg import FedAvgServer
from fedlerate.clients.client_ditto import DittoClient

class DittoServer(FedAvgServer):
    def __init__(self, config):
        super(DittoServer, self).__init__(config)

    def init_clients(self):
        # Initialize clients based on the configuration
        self.clients = self._init_clients_parallel(DittoClient)