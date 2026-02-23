"""
fedlerate
---------

fedlerate is a Python library that provides building blocks for federated learning systems. It offers a modular and extensible framework for creating and managing clients and servers in a federated learning setup. The library includes base classes for clients and servers, as well as utilities for training, evaluation, and data partitioning.
"""

from fedlerate.clients.client_base import BaseClient
from fedlerate.clients.client_ditto import DittoClient
from fedlerate.clients.client_fedprox import FedProxClient
from fedlerate.clients.client_pfl import PFLClient
from fedlerate.servers.server_fedavg import FedAvgServer
from fedlerate.servers.server_fedper import FedPerServer
from fedlerate.servers.server_fedprox import FedProxServer
from fedlerate.servers.server_ditto import DittoServer


__all__ = ['BaseClient',
            'DittoClient', 
            'FedProxClient', 
            'PFLClient', 
            'FedAvgServer',
            'FedProxServer',
            'DittoServer',
            'FedPerServer']
__version__ = "0.1.0"