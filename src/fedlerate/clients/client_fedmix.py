import gc
import torch
from fedlerate.clients.client_abc import ClientABC
from fedlerate.data.partition import load_partition
from fedlerate.models.functions import evaluate_fedmix, train_fedmix
from fedlerate.models.models import load_model


class FedMixClient(ClientABC):

    def __init__(self, client_id, config, device):
        super(FedMixClient, self).__init__(client_id, config, device)
        self.init_model(k=self.config["n_clusters"])
        self.init_data()
        self.weights = torch.nn.Parameter(
            torch.ones(self.config["n_clusters"], device=self.device)
            / self.config["n_clusters"]
        )

    def init_model(self, k=None):
        """Allocate the k mixture models once and keep them alive across rounds."""
        n_clusters = self.config["n_clusters"]
        self.models = [
            load_model(self.config["model_name"], n_classes=self.config["n_classes"], device=self.device)
            for _ in range(n_clusters)
        ]

    def init_data(self):
        self.trainloader, self.testloader = self._get_dataloaders()

    def _get_dataloaders(self):
        # Fetch dataloaders strictly when needed
        trainloader, testloader, _ = load_partition(
            num_partition=self.client_id, batch_size=self.config["batch_size"]
        )
        return trainloader, testloader

    def train(self, global_weights=None, budget=1.0):
        if global_weights is None:
            raise ValueError(
                "FedMixClient requires global_weights passed to train()."
            )

        data_size = len(self.trainloader.dataset)

        # 2. Reload weights into the cached model instances (no allocation)
        for i, w in enumerate(global_weights):
            self.models[i].load_state_dict(w)

        # 3. Train
        loss, _ = train_fedmix(
            self.models,
            self.weights,
            dataloader=self.trainloader,
            epochs=self.config["local_epochs"],
            lr=self.config["local_lr"],
            device=self.device,
            noise_mixer=getattr(self, "noise_mixer", None),
            budget=budget
        )

        self.config["local_lr"] *= self.config.get("lr_decay", 1.0)

        # 4. Extract weights
        updated_weights = [
            {k: v.cpu().clone() for k, v in m.state_dict().items()}
            for m in self.models
        ]

        return {
            "weights": updated_weights,
            "data_size": data_size,
            "loss": loss,
        }

    def evaluate(self, global_weights=None):
        if global_weights is None:
            raise ValueError(
                "FedMixClient requires global_weights passed to evaluate()."
            )

        # Reload weights into the cached model instances (no allocation)
        for i, w in enumerate(global_weights):
            self.models[i].load_state_dict(w)

        accuracy = evaluate_fedmix(
            self.models,
            self.weights,
            dataloader=self.testloader,
            device=self.device,
        )

        return accuracy