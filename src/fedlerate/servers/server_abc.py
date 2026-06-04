from abc import ABC, abstractmethod
import logging
import csv
import os
from concurrent.futures import ThreadPoolExecutor

import torch


class ServerABC(ABC):
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.init_clients()
        self.init_model()
        self.strategy = config['strategy']

        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger(self.__class__.__module__)

    def _init_clients_parallel(self, client_class):
        n_jobs = self.config.get('n_jobs', 1)
        if n_jobs > 1:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                return list(executor.map(lambda i: client_class(client_id=i, config=self.config, device=self.device), range(self.config['n_clients'])))
        else:
            return [client_class(client_id=i, config=self.config, device=self.device) for i in range(self.config['n_clients'])]

    def _broadcast_parallel(self, clients, weights):
        n_jobs = self.config.get('n_jobs', 1)
        if n_jobs > 1:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                list(executor.map(lambda c: c.set_model_weights(weights), clients))
        else:
            for client in clients:
                client.set_model_weights(weights)

    def _save_metrics(self, metrics: dict, file_path: str):
        # Check if file exists to decide whether to write header
        file_exists = os.path.isfile(file_path)

        with open(file_path, mode="a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=metrics.keys()
            )

            # Write header only once
            if not file_exists:
                writer.writeheader()

            writer.writerow(metrics)

    def _log_metrics(self, metrics: dict):
        accuracy_str = '{:.3f}'.format(metrics['accuracy']) if metrics['accuracy'] is not None else "N/A"
        loss_str = '{:.3f}'.format(metrics['loss']) if metrics['loss'] is not None else "N/A"

        if self.config["model_name"] == "PFL_AutoEncoder":
            self.log.info(f"Round {metrics['round']} - Avg validation F1: {accuracy_str} - Avg training loss: {loss_str}")
        else:
            self.log.info(f"Avg accuracy: {accuracy_str} - Avg loss: {loss_str}")

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def aggregate(self, client_updates: list[dict]) -> None:
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def broadcast_model(self):
        pass

    @abstractmethod
    def init_clients(self):
        pass

    @abstractmethod
    def select_clients(self):
        pass

    @abstractmethod
    def init_model(self):
        pass