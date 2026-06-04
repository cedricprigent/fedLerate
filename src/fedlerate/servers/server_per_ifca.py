from fedlerate.servers.server_ifca import IFCAServer
from fedlerate.clients.client_pfl import PFLClient
from fedlerate.models.models import load_model


class PerIFCAServer(IFCAServer):
    def __init__(self, config):
        super(PerIFCAServer, self).__init__(config)

    def init_clients(self):
        self.clients = self._init_clients_parallel(PFLClient)

    def init_model(self) -> None:
        self.models = [load_model(self.config['model_name'], device='cpu').global_layers for _ in range(self.config['n_clusters'])]
