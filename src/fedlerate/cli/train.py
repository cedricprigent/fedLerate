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
    
    strategy_keys = ["fedprox_mu", "ditto_lambda", "n_clusters", "n_jobs"]
    for k in strategy_keys:
        if k not in config:
            config[k] = None
    
    # Ensure n_jobs is at least 1 and not None
    if config.n_jobs is None:
        config.n_jobs = 1
    
    print(f'### Starting {strategy} on {dataset} with n_jobs={config.n_jobs} ###')

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
        n_jobs=config.n_jobs,
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
    if strategy in ['FCFL', 'PerFCFL']:
        # Hydra changes the CWD to the run directory. 
        # We need to look 4 levels up to reach the MPC project root.
        root_prefix = Path('../../../../')
        
        # Check for best_k_path in config
        if 'best_k_path' in config and config.best_k_path:
            best_k_path = root_prefix / config.best_k_path
            if best_k_path.exists():
                with open(best_k_path, 'r') as f:
                    d['n_clusters'] = int(f.read().strip())
        
        # Fallback to current/best_k.txt at root if n_clusters still not set
        if d['n_clusters'] is None:
            fallback_k = root_prefix / 'current/best_k.txt'
            if fallback_k.exists():
                 with open(fallback_k, 'r') as f:
                    d['n_clusters'] = int(f.read().strip())

        # Load labels
        labels_npy_path = root_prefix / 'current/labels.npy'
        if labels_npy_path.exists():
            labels = np.load(labels_npy_path)
        else:
            labels = np.load(root_prefix / config.labels_path)
            
        with open(root_prefix / config.config_clustering_path, 'r') as f:
            config_clustering = json.load(f)
        
        d.update({"strategy": f"{strategy}-{config_clustering['clustering']['distance_metric']}-{config_clustering['protocol']['domain_size']}"})
        d.update({"labels": labels})
        if d['n_clusters'] is None:
            d['n_clusters'] = len(np.unique(labels))
        
        if strategy == 'FCFL':
            server = FCFLServer(d)
        else:
            server = PerFCFLServer(d)
        server.run()
    if strategy == 'FedMix':
        server = FedMixServer(d)
        server.run()

if __name__ == '__main__':
    main()