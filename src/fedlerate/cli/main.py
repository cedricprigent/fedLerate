import argparse

from fedlerate.datasets.mnist_partitions import generate_mnist_partitions
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
        help="Dataset name"
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
    
    if args.dataset == "MNIST":
        generate_mnist_partitions()
    if args.dataset == "PACS-soft":
        generate_pacs_soft_partitions()
    if args.dataset == "PACS-strct":
        generate_pacs_strict_partitions()

if __name__ == "__main__":
    main()