from abc import ABC, abstractmethod
import logging
import csv
import os

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
        if self.config["model_name"] == "PFL_AutoEncoder":
            self.log.info(f"Round {metrics['round']} - Avg validation F1: {'{:.6f}'.format(metrics['accuracy'])} - Avg training loss: {'{:.6f}'.format(metrics['loss'])}")
        else:
            self.log.info(f"Avg accuracy: {'{:.3f}'.format(metrics['accuracy'])} - Avg loss: {'{:.3f}'.format(metrics['loss'])}")

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