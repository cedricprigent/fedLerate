import copy
import torch
from fedlerate.clients.client_base import BaseClient
from fedlerate.models.functions import train

class RDPCFLClient(BaseClient):
    def __init__(self, client_id, config, device):
        super(RDPCFLClient, self).__init__(client_id, config, device)
        self.clip_norm = self.config.get('clip_norm', 1.0)
        self.noise_multiplier = self.config.get('noise_multiplier', 0.5)

    def extract_flat_params(self):
        """Flattens all trainable parameters of self.model into a single 1D tensor."""
        return torch.cat([p.detach().clone().view(-1) for p in self.model.parameters()])

    def train(self):
        # Save global model parameters before local training
        flat_global = self.extract_flat_params()

        # Local training using the framework's train function
        loss, _ = train(
            self.model,
            dataloader=self.trainloader,
            epochs=self.config['local_epochs'],
            lr=self.config['local_lr'],
            device=self.device,
            noise_mixer=self.noise_mixer
        )

        self.config['local_lr'] *= self.config.get('lr_decay', 1.0)

        # Compute difference (Update Vector: Delta = Theta_local - Theta_global)
        flat_local = self.extract_flat_params()
        update_vector = flat_local - flat_global

        # 1. Differential Privacy: Gradient/Update Clipping
        update_norm = torch.norm(update_vector, p=2)
        clip_factor = max(1.0, float(update_norm.item() / self.clip_norm))
        clipped_update = update_vector / clip_factor

        # 2. Differential Privacy: Gaussian Noise Addition
        sigma = self.noise_multiplier * self.clip_norm
        noise = torch.normal(mean=0.0, std=sigma, size=clipped_update.shape, device=clipped_update.device)
        dp_update = clipped_update + noise

        return {
            "weights": self.get_model_weights(),
            "dp_update": dp_update.cpu(),
            "data_size": len(self.trainloader.dataset),
            "loss": loss
        }

    def evaluate_loss(self, model_weights):
        """Evaluate training loss on a given set of weights (for Loss Refinement)."""
        try:
            sampling_ratio = self.config.get('sampling_ratio')
        except:
            raise ValueError("sampling_ratio is not specified in the config.")

        old_weights = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(model_weights)
        self.model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        samples = 0

        n_batches = len(self.trainloader)
        n_sampled_batches = max(1, int(n_batches * sampling_ratio))

        with torch.no_grad():
            for i, (x, y) in enumerate(self.trainloader):
                if i >= n_sampled_batches:
                    break

                if len(x.shape) == 3:
                    x = x.unsqueeze(1)
                x = x.to(self.device)
                y = y.to(self.device)
                out = self.model(x)
                loss = criterion(out, y)
                total_loss += loss.item() * len(y)
                samples += len(y)

        self.model.load_state_dict(old_weights)
        return total_loss / max(1, samples)
