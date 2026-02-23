from fedlerate import FedAvgServer

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
        strategy=strategy
    )

    # Run experiment
    if strategy == 'FedAvg':
        server = FedAvgServer(d)
        server.run()


if __name__ == '__main__':
    app()