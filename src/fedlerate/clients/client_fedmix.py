import gc
import torch
from fedlerate.clients.client_abc import ClientABC
from fedlerate.data.partition import load_partition
from fedlerate.models.functions import evaluate_fedmix, train_fedmix
from fedlerate.models.models import load_model


class FedMixClient(ClientABC):

    def __init__(self, client_id, config, device):
        self.models = None
        super(FedMixClient, self).__init__(client_id, config, device)
        self.weights = torch.nn.Parameter(
            torch.ones(self.config["n_clusters"], device=self.device)
            / self.config["n_clusters"]
        )

    def init_model(self, k=None):
        self.models = None

    def init_data(self):
        # Do NOTHING here! Don't load data during server init.
        pass

    def _get_dataloaders(self):
        # Fetch dataloaders strictly when needed
        trainloader, testloader, _ = load_partition(
            num_partition=self.client_id, batch_size=self.config["batch_size"]
        )
        return trainloader, testloader

    def train(self, global_weights=None):
        if global_weights is None:
            raise ValueError(
                "FedMixClient requires global_weights passed to train()."
            )

        # 1. Instantiate DataLoaders on-demand
        trainloader, _ = self._get_dataloaders()
        data_size = len(trainloader.dataset)

        # 2. Instantiate models on-demand
        k = self.config["n_clusters"]
        models = [
            load_model(self.config["model_name"], n_classes=self.config["n_classes"], device=self.device)
            for _ in range(k)
        ]
        for i, w in enumerate(global_weights):
            models[i].load_state_dict(w)

        # 3. Train
        loss, _ = train_fedmix(
            models,
            self.weights,
            dataloader=trainloader,
            epochs=self.config["local_epochs"],
            lr=self.config["local_lr"],
            device=self.device,
            noise_mixer=getattr(self, "noise_mixer", None),
        )

        self.config["local_lr"] *= self.config.get("lr_decay", 1.0)

        # 4. Extract weights
        updated_weights = [
            {k: v.cpu().clone() for k, v in m.state_dict().items()}
            for m in models
        ]

        # 5. Full memory purge: delete dataloader, models, clear cache
        del trainloader
        del models
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

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

        _, testloader = self._get_dataloaders()

        k = self.config["n_clusters"]
        models = [
            load_model(self.config["model_name"], n_classes=self.config["n_classes"], device=self.device)
            for _ in range(k)
        ]
        for i, w in enumerate(global_weights):
            models[i].load_state_dict(w)

        accuracy = evaluate_fedmix(
            models,
            self.weights,
            dataloader=testloader,
            device=self.device,
        )

        # Full memory purge
        del testloader
        del models
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return accuracy