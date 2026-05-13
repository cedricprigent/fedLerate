### FedLerate Installation

```bash
pip install -e .
```

### FedLerate Partition

FedLerate provides utils to automatically generate client partitions for data drift scenarios.

```bash
fedlerate-partition -d MNIST -a 10 -c 200
fedlerate-partition -d DigitsDrift -a 1 -c 200
fedlerate-partition -d PACS-strict -a 10 -c 40
```

### FedLerate Experiment

```bash
fedlerate-train --config-dir $config_directory --config-name $config_name
```