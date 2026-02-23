from fedlerate import FedAvgServer
import pytest
import torch


def test_fedavg_server():
    # Example configuration for testing
    config = {
        'n_clients': 5,
        'n_clients_per_round': 3,
        'local_epochs': 1,
        'batch_size': 16,
        'model_name': 'DummyModel',
        'strategy': 'fedavg',
    }

    # Initialize FedAvg server
    server = FedAvgServer(config)

    # Check model initialization
    server.init_model()
    assert server.global_model is not None

    # Check client initialization
    server.init_clients()
    assert len(server.clients) == config['n_clients']

    # Check model broadcasting
    server.broadcast_model()
    assert torch.equal(next(server.clients[0].model.parameters()).cpu(), next(server.global_model.parameters()).cpu())

    # Check client selection
    selected_clients = server.select_clients()
    assert len(selected_clients) == config['n_clients_per_round']

    # Check aggregation logic with dummy updates
    fake_client_updates = [{
        'weights': {key: torch.ones_like(param)*4 for key, param in server.global_model.state_dict().items()},
        'data_size': 10
    }, {
        'weights': {key: torch.ones_like(param) for key, param in server.global_model.state_dict().items()},
        'data_size': 20
    }]
    server.aggregate(fake_client_updates)
    assert server.global_model.state_dict() == {key: torch.ones_like(param)*2 for key, param in server.global_model.state_dict().items()}


if __name__ == "__main__":
    pytest.main([__file__])