"""
FLStratum
---------

FLStratum is a Python library that provides building blocks for federated learning systems. It offers a modular and extensible framework for creating and managing clients and servers in a federated learning setup. The library includes base classes for clients and servers, as well as utilities for training, evaluation, and data partitioning.
"""

from flstratum.clients.client_base import BaseClient
from flstratum.clients.client_ditto import DittoClient
from flstratum.clients.client_fedprox import FedProxClient
from flstratum.clients.client_pfl import PFLClient
from flstratum.servers.server_fedavg import FedAvgServer
from flstratum.servers.server_fedper import FedPerServer
from flstratum.servers.server_fedprox import FedProxServer
from flstratum.servers.server_ditto import DittoServer


__all__ = ['BaseClient',
            'DittoClient', 
            'FedProxClient', 
            'PFLClient', 
            'FedAvgServer',
            'FedProxServer',
            'DittoServer',
            'FedPerServer']
__version__ = "0.1.0"