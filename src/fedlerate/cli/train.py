from fedlerate.servers.server_fedavg import FedAvgServer
from fedlerate.servers.server_fedprox import FedProxServer
from fedlerate.servers.server_fedper import FedPerServer
from fedlerate.servers.server_ditto import DittoServer
from fedlerate.servers.server_ifca import IFCAServer
from fedlerate.servers.server_fcfl import FCFLServer
from fedlerate.servers.server_per_ifca import PerIFCAServer
from fedlerate.servers.server_per_fcfl import PerFCFLServer
from fedlerate.servers.server_fedmix import FedMixServer

import hydra
from pathlib import Path
import numpy as np
import json

from omegaconf import OmegaConf

# python main.py --config-dir conf --config_name config
@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config):

    # Hydra changes working directory to run dir
    run_dir = Path.cwd()
    metrics_save_path = run_dir / "metrics.csv"

    # Extract config parameters
    strategy = config.strategy
    dataset = config.scenario.dataset

    OmegaConf.set_struct(config, False)
    
    strategy_keys = ["fedprox_mu", "ditto_lambda", "n_clusters"]
    for k in strategy_keys:
        if k not in config:
            config[k] = None

    # Base parameters
    d = dict(
        n_rounds=config.scenario.n_rounds,
        n_clients=config.scenario.n_clients,
        local_epochs=config.scenario.local_epochs,
        model_name=config.scenario.model,
        n_clients_per_round=config.scenario.n_clients_per_round,
        batch_size=config.scenario.batch_size,
        metrics_save_path=metrics_save_path,
        strategy=strategy,
        local_lr=config.scenario.local_lr,
        fedprox_mu=config.fedprox_mu,
        ditto_lambda=config.ditto_lambda,
        n_clusters=config.n_clusters,
    )

    # Run experiment
    if strategy == 'FedAvg':
        server = FedAvgServer(d)
        server.run()
    if strategy == 'FedProx':
        server = FedProxServer(d)
        server.run()
    if strategy == 'FedPer':
        server = FedPerServer(d)
        server.run()
    if strategy == 'Ditto':
        server = DittoServer(d)
        server.run()
    if strategy == 'IFCA':
        server = IFCAServer(d)
        server.run()
    if strategy == 'PerIFCA':
        server = PerIFCAServer(d)
        server.run()
    if strategy == 'FCFL':
        labels = np.load('../../../../' +config.labels_path)
        with open('../../../../' + config.config_clustering_path, 'r') as f:
            config_clustering = json.load(f)
        
        d.update({"strategy": f"FCFL-{config_clustering['clustering']['distance_metric']}-{config_clustering['protocol']['domain_size']}"})
        d.update({"labels":labels})
        
        server = FCFLServer(d)
        server.run()
    if strategy == 'PerFCFL':
        labels = np.load('../../../../' +config.labels_path)
        with open('../../../../' + config.config_clustering_path, 'r') as f:
            config_clustering = json.load(f)
        
        d.update({"strategy": f"PerFCFL-{config_clustering['clustering']['distance_metric']}-{config_clustering['protocol']['domain_size']}"})
        d.update({"labels":labels})

        server = PerFCFLServer(d)
        server.run()
    if strategy == 'FedMix':
        server = FedMixServer(d)
        server.run()

if __name__ == '__main__':
    main()