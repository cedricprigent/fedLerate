from fedlerate.servers.server_fcfl import FCFLServer
from fedlerate.clients.client_pfl import PFLClient
from fedlerate.models.models import load_model


class PerFCFLServer(FCFLServer):
    def __init__(self, config):
        super(PerFCFLServer, self).__init__(config)

    def init_clients(self):
        self.clients = [PFLClient(client_id=i, config=self.config, device=self.device) for i in range(self.config['n_clients'])]

    def init_model(self) -> None:
        self.models = [load_model(self.config['model_name'], device='cpu').global_layers for _ in range(self.config['n_clusters'])]
