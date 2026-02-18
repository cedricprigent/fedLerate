"""
FLBase
-------

FLBase is a Python library that provides building blocks for federated learning systems. It offers a modular and extensible framework for creating and managing clients and servers in a federated learning setup. The library includes base classes for clients and servers, as well as utilities for training, evaluation, and data partitioning.
"""

from flbase.clients.client_base import BaseClient
from flbase.servers.server_fedavg import FedAvgServer


__all__ = ['BaseClient', 'FedAvgServer']
__version__ = "0.1.0"