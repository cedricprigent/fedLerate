from fedlerate.servers.server_fedavg import FedAvgServer
from fedlerate.servers.server_fedprox import FedProxServer
from fedlerate.servers.server_fedper import FedPerServer
from fedlerate.servers.server_ditto import DittoServer
from fedlerate.servers.server_ifca import IFCAServer
from fedlerate.servers.server_fcfl import FCFLServer
from fedlerate.servers.server_fpercfl import FPerCFLServer
from fedlerate.servers.server_fedmix import FedMixServer

import hydra
from pathlib import Path
import numpy as np

# python app.py --config-path conf --config_name config
@hydra.main(config_path='config', config_name='config', version_base=None)
def app(config):

    # Hydra changes working directory to run dir
    run_dir = Path.cwd()
    metrics_save_path = run_dir / "metrics.csv"

    # Extract config parameters
    strategy = config.strategy
    dataset = config.scenario.dataset

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
        fedprox_mu=config.fedprox_mu if strategy in ['FedProx'] else None,
        ditto_lambda=config.ditto_lambda if strategy in ['Ditto'] else None,
        n_clusters=config.n_clusters if strategy in ['IFCA', 'FCFL', 'FedMix'] else None,
        labels=config.labels if strategy in ['FCFL'] else None
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
    if strategy == 'FCFL':
        server = FCFLServer(d)
        server.run()
    if strategy == 'FPerCFL':
        server = FPerCFLServer(d)
        server.run()
    if strategy == 'FedMix':
        server = FedMixServer(d)
        server.run()

if __name__ == '__main__':
    app()