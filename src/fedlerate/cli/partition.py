import argparse
import random

from fedlerate.datasets.mnist_partitions import generate_mnist_partitions
from fedlerate.datasets.digits_drift_partitions import create_digits_drift_partitions
from fedlerate.datasets.pacs_soft_partitions import generate_pacs_mixed_partitions
from fedlerate.datasets.pacs_strict_partitions import generate_pacs_strict_partitions
from fedlerate.datasets.speech_commands import generate_speech_commands_partitions
from fedlerate.datasets.femnist import create_femnist_partitions
from fedlerate.datasets.officehome import generate_office_home_strict

def build_parser():
    parser = argparse.ArgumentParser(
        prog="fedlerate",
        description="fedLerate CLI"
    )

    parser.add_argument(
        "-d", "--dataset",
        required=True,
        type=str,
        help="Dataset name"
    )

    parser.add_argument(
        "-a", "--alpha",
        required=True,
        type=float,
        help="Concentration parameter"
    )

    parser.add_argument(
        "-c", "--clients",
        required=True,
        type=int,
        help="Number of clients/partitions"
    )

    parser.add_argument(
        "-s", "--seed",
        type=int,
        help="Set random seed for generation"
    )

    parser.add_argument(
        "--root",
        type=str,
        default='~/datasets',
        help="Root directory for the dataset"
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
    root = args.root
    random.seed(args.seed)
    
    if dataset == "MNIST":
        generate_mnist_partitions(alpha=alpha, n_clients=clients)
    elif dataset == "DigitsDrift":
        create_digits_drift_partitions(alpha=alpha, n_clients=clients)
    elif dataset == "PACS-soft":
        generate_pacs_mixed_partitions(alpha=alpha, total_clients=clients, root=root)
    elif dataset == "PACS-strict":
        generate_pacs_strict_partitions(alpha=alpha, n_partitions_per_domain=clients//4, root=root)
    elif dataset == "SpeechCommands":
        generate_speech_commands_partitions(alpha=alpha, n_clients=clients, root=root)
    elif dataset == "FEMNIST":
        create_femnist_partitions(n_clients=clients, data_dir=root)
    elif dataset == "OfficeHome":
        generate_office_home_strict(raw_dir=root)

if __name__ == "__main__":
    main()