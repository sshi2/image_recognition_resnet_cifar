import argparse
import configparser

from train import train


def get_arguments():
    """args function"""

    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--epoch", type=int, default=100)
    parser.add_argument("-m", "--model", type=int,
                        choices=[18, 34, 50, 101, 152], default=18)
    parser.add_argument("-lr", "--learning_rate", type=int, default=0.001)
    parser.add_argument("-mt", "--momentum", type=int, default=0.9)
    parser.add_argument("-d", "--dataset", type=str,
                        choices=["CIFAR10", "CIFAR100", "MNIST"], default="CIFAR10")
    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument("-nw", "--num_workers", type=int, default=1)

    return parser.parse_args()


def main():
    """
    main function.
    This function performs all of program
    """

    # get args
    args = get_arguments()

    # load config
    config = configparser.ConfigParser()
    config.read("./config.ini")

    # train
    train(args, config)


if __name__ == '__main__':
    main()
