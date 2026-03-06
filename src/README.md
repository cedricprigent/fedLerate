## fedLerate: FL Simulation Framework

### Using fedLerate CLI
```sh
# Generate rotated MNIST partitions
python -m fedlerate.cli.main partition -d MNIST

# Generate PACS domain drift partitions
python -m fedlerate.cli.main partition -d PACS-strict
```