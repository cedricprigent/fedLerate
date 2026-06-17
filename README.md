# FedLerate

[![Homepage](https://img.shields.io/badge/Homepage-GitHub-blue)](https://github.com/cedricprigent/fedlerate)

Building blocks for federated learning systems.

## Installation

```bash
pip install -e .
```

## Data Partitioning

`fedlerate-partition` provides utilities to automatically generate client partitions for federated learning, including data drift scenarios.

### Usage

```bash
fedlerate-partition -d <dataset> -a <alpha> -c <clients> [--root <root_dir>] [-s <seed>]
```

### Options

- `-d`, `--dataset`: Dataset name (e.g., `MNIST`, `DigitsDrift`, `PACS-soft`, `PACS-strict`, `SpeechCommands`).
- `-a`, `--alpha`: Concentration parameter for Dirichlet distribution (controls heterogeneity).
- `-c`, `--clients`: Number of clients/partitions.
- `--root`: Root directory for the dataset (default: `~/datasets`).
- `-s`, `--seed`: Random seed for generation.

### Examples

```bash
# Partition MNIST into 200 clients with alpha=10
fedlerate-partition -d MNIST -a 10 -c 200

# Partition SpeechCommands with a specific root directory
fedlerate-partition -d SpeechCommands --root ~/datasets -c 200 -a 1

# Partition PACS-strict
fedlerate-partition -d PACS-strict -a 10 -c 40
```

## Federated Training

`fedlerate-train` is the main entry point for running federated learning experiments. It is powered by [Hydra](https://hydra.cc), allowing for flexible configuration.

### Usage

```bash
fedlerate-train --config-dir <config_directory> --config-name <config_name> [hydra_overrides]
```

### Supported Strategies

- `fedavg`: [Federated Averaging](https://arxiv.org/pdf/1602.05629)
- `fedprox`: [Federated Optimization with Proximal term](https://arxiv.org/pdf/1812.06127)
- `fedper`: [Federated Learning with Personalization Layers](https://arxiv.org/pdf/1912.00818)
- `ditto`: [Fair and Robust Federated Learning Through Personalization](https://arxiv.org/pdf/2012.04221)
- `ifca`: [Iterative Federated Clustering Algorithm](https://arxiv.org/pdf/2006.04088)
- `fcfl`: [Feature Clustering based Federated Learning](https://hal.science/hal-04779813/file/Efficient_Resource_Constrained_FL_Clustering.pdf)
- `fedmix`: [Federated Mixture of Experts](https://arxiv.org/pdf/2107.06724)

### Example

```bash
fedlerate-train --config-dir configs/speech --config-name fedper hydra.run.dir=logs/speech/fedper/${now:%Y-%m-%d}-${now:%H-%M-%S}
```

### Configuration

Experiments are configured using YAML files. You can find examples in the `tests/configs` directory.

```bash
# Run FedAvg on MNIST (assuming config exists)
fedlerate-train --config-dir tests/configs/mnist --config-name fedavg
```