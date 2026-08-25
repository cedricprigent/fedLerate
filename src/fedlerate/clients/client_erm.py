from fedlerate.clients.client_base import BaseClient
from fedlerate.models.functions import evaluate_erm


class ClientERM(BaseClient):
    def __init__(self, client_id, config, device):
        super(ClientERM, self).__init__(client_id, config, device)

    def evaluate_erm(self):
        try:
            sampling_ratio = self.config.get('sampling_ratio')
        except:
            raise ValueError("sampling_ratio is not specified in the config.")

        # Evaluate local model on client test data
        accuracy = evaluate_erm(self.model, 
            dataloader=self.trainloader, 
            sampling_ratio=sampling_ratio,
            device=self.device
        )
        return accuracy