from abc import ABC, abstractmethod
from fedlerate.models.models import load_model

class ClientABC(ABC):
    def __init__(self, client_id, config, device):
        self.client_id = client_id
        self.config = config        
        self.device = device
        if self.config.get('use_noise_mixer', False):
            self.load_noise_mixer()
        else:
            self.noise_mixer = None

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

    def load_noise_mixer(self):
        from fedlerate.models.functions import get_background_noise_list, BackgroundNoiseMixer
        noise_list = get_background_noise_list()
        self.noise_mixer = BackgroundNoiseMixer(noise_list)