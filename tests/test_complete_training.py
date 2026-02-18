import torch
from flstratum.servers.server_fedavg import FedAvgServer


def test_complete_training():
    # Example configuration for testing
    config = {
        'n_rounds': 2,
        'n_clients': 5,
        'n_clients_per_round': 3,
        'local_epochs': 1,
        'batch_size': 16,
        'model_name': 'LeNet5',
        'strategy': 'fedavg',
        'metrics_save_path': 'test_metrics.json',
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize FedAvg server
    server = FedAvgServer(config)

    # Run the complete training process
    server.run()

    # Check if the global model has been updated after training
    assert server.global_model is not None


if __name__ == "__main__":
    test_complete_training()