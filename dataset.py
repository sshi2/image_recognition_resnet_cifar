import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    # 複数のtransformを連続して行う
    Compose,
    # 自作の関数を渡すことで実行可能
    Lambda,
    # ランダムに画像を切り抜く
    RandomCrop,
    # ランダムに左右反転を行う
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    MNIST
)


def cifar10():
    '''
    load dataset from torchvision.
    '''

    train_transform = transforms.Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
    ])

    val_transform = transforms.Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
    ])

    train_dataset = CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    val_dataset = CIFAR10(
        root='./data', train=False, download=True, transform=val_transform
    )

    return train_dataset, val_dataset


def cifar100():
    '''
    load dataset from torchvision.
    '''

    train_transform = transforms.Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
    ])

    val_transform = transforms.Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
    ])

    train_dataset = CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    val_dataset = CIFAR100(
        root='./data', train=False, download=True, transform=val_transform
    )

    return train_dataset, val_dataset


def load_dataset(args):
    '''
    This function performs to load dataset.
    If you set args.dataset, you can select which datasets you use.
    '''
    if args.dataset == "CIFAR10":
        train_dataset, val_dataset = cifar10()
    elif args.dataset == "CIFAR100":
        train_dataset, val_dataset = cifar100()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
        shuffle=True
    )

    return train_loader, val_loader
