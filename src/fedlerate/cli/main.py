import argparse

from fedlerate.datasets.mnist_partitions import generate_mnist_partitions
from fedlerate.datasets.digits_drift_partitions import create_digits_drift_partitions
from fedlerate.datasets.pacs_soft_partitions import generate_pacs_soft_partitions
from fedlerate.datasets.pacs_strict_partitions import generate_pacs_strict_partitions

def build_parser():
    parser = argparse.ArgumentParser(
        prog="fedlerate",
        description="fedLerate CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    # ---- partition command ----
    partition_parser = subparsers.add_parser(
        "partition",
        help="Generate client partitions"
    )

    partition_parser.add_argument(
        "-d", "--dataset",
        required=True,
        type=str,
        help="Dataset name"
    )

    partition_parser.add_argument(
        "-a", "--alpha",
        required=True,
        type=float,
        help="Concentration parameter"
    )

    partition_parser.add_argument(
        "-c", "--clients",
        required=True,
        type=int,
        help="Number of clients/partitions"
    )

    return parser

def main():
    """
    Calls partition generation script
    :param d: dataset

    ### Example
        >>> python -m fedlerate.cli.main partition -d MNIST
    """


    parser = build_parser()
    args = parser.parse_args()
    dataset = args.dataset
    alpha = args.alpha
    clients = args.clients
    
    if dataset == "MNIST":
        generate_mnist_partitions(alpha=alpha, n_clients=clients)
    elif dataset == "DigitsDrift":
        create_digits_drift_partitions(alpha=alpha, n_clients=clients)
    elif dataset == "PACS-soft":
        generate_pacs_soft_partitions()
    elif dataset == "PACS-strict":
        generate_pacs_strict_partitions()

if __name__ == "__main__":
    main()