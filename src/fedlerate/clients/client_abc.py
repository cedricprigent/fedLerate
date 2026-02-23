from abc import ABC, abstractmethod
from fedlerate.models.models import load_model

class ClientABC(ABC):
    def __init__(self, client_id, config, device):
        self.client_id = client_id
        self.config = config        
        self.device = device

    def set_model_weights(self, weights):
        if self.model is not None:
            self.model.load_state_dict(weights)

    def get_model_weights(self):
        if self.model is not None:
            return self.model.state_dict()
        else:
            return None

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def init_data(self):
        pass