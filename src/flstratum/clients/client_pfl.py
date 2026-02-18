from flstratum.clients.client_base import BaseClient
from flstratum.models.functions import train, evaluate
from flstratum.models.models import load_model


class PFLClient(BaseClient):
    def __init__(self, client_id, config, device):
        super(PFLClient, self).__init__(client_id, config, device)

    def init_model(self):
        self.model = load_model(self.config['model_name'], device=self.device)

    def set_model_weights(self, weights):
        self.model.global_layers.load_state_dict(weights)

    def get_model_weights(self):
        if self.model.global_layers is not None:
            return self.model.global_layers.state_dict()
        else:
            return None