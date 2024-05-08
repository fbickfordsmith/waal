from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, SVHN, FashionMNIST
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor


Array = Union[np.ndarray, Tensor]


dataset_configs = {
    "FashionMNIST": {
        "train_transform": Compose(
            [
                ToTensor(),
                Normalize((0.1307,), (0.3081,)),
            ]
        ),
        "test_transform": Compose(
            [
                ToTensor(),
                Normalize((0.1307,), (0.3081,)),
            ]
        ),
        "train_loader_config": {"batch_size": 64, "num_workers": 1},
        "test_loader_config": {"batch_size": 1_000, "num_workers": 1},
        "optimizer_config": {"lr": 0.01, "momentum": 0.5},
    },
    "SVHN": {
        "train_transform": Compose(
            [
                RandomCrop(size=32, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ]
        ),
        "test_transform": Compose(
            [
                ToTensor(),
                Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ]
        ),
        "train_loader_config": {"batch_size": 64, "num_workers": 1},
        "test_loader_config": {"batch_size": 1_000, "num_workers": 1},
        "optimizer_config": {"lr": 0.01, "momentum": 0.5},
    },
    "CIFAR10": {
        "train_transform": Compose(
            [
                RandomCrop(size=32, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        ),
        "test_transform": Compose(
            [
                ToTensor(),
                Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        ),
        "train_loader_config": {"batch_size": 64, "num_workers": 1},
        "test_loader_config": {"batch_size": 1_000, "num_workers": 1},
        "optimizer_config": {"lr": 0.01, "momentum": 0.3},
    },
}


@dataclass
class TrainDataset(Dataset):
    inputs_0: Array
    labels_0: Array
    inputs_1: Array
    labels_1: Array
    preprocess: Callable = Image.fromarray
    transform: Callable = None

    def __len__(self) -> int:
        return max(len(self.inputs_0), len(self.inputs_1))

    def __getitem__(self, index: int) -> Tuple[Array, Array, Array, Array]:
        index_0 = index if index < len(self.labels_0) else index % len(self.inputs_0)
        index_1 = index if index < len(self.labels_1) else index % len(self.inputs_1)

        inputs_0 = self.inputs_0[index_0]
        inputs_1 = self.inputs_1[index_1]

        labels_0 = self.labels_0[index_0]
        labels_1 = self.labels_1[index_1]

        if self.transform != None:
            inputs_0 = self.preprocess(inputs_0)
            inputs_1 = self.preprocess(inputs_1)

            inputs_0 = self.transform(inputs_0)
            inputs_1 = self.transform(inputs_1)

        return inputs_0, labels_0, inputs_1, labels_1


@dataclass
class TestDataset(Dataset):
    inputs: Array
    labels: Array
    preprocess: Callable = Image.fromarray
    transform: Callable = None

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> Tuple[Array, Array]:
        inputs, labels = self.inputs[index], self.labels[index]

        if self.transform != None:
            inputs = self.preprocess(inputs)
            inputs = self.transform(inputs)

        return inputs, labels


def get_cifar10(data_dir: Path) -> Tuple[np.ndarray, Tensor, np.ndarray, Tensor]:
    train_dataset = CIFAR10(data_dir / "CIFAR10", train=True, download=True)
    test_dataset = CIFAR10(data_dir / "CIFAR10", train=False, download=True)

    train_inputs = train_dataset.data  # np.uint8, [50_000, 32, 32, 3]
    test_inputs = test_dataset.data  # np.uint8, [10_000, 32, 32, 3]

    train_labels = torch.tensor(train_dataset.targets)  # torch.int64, [50_000,]
    test_labels = torch.tensor(test_dataset.targets)  # torch.int64, [10_000,]

    return train_inputs, train_labels, test_inputs, test_labels


def get_fashionmnist(data_dir: Path) -> Tuple[np.ndarray, Tensor, np.ndarray, Tensor]:
    train_dataset = FashionMNIST(data_dir / "FashionMNIST", train=True, download=True)
    test_dataset = FashionMNIST(data_dir / "FashionMNIST", train=False, download=True)

    train_inputs = train_dataset.data.numpy()  # np.uint8, [60_000, 28, 28]
    test_inputs = test_dataset.data.numpy()  # np.uint8, [10_000, 28, 28]

    train_labels = train_dataset.targets  # torch.int64, [60_000,]
    test_labels = test_dataset.targets  # torch.int64, [10_000,]

    return train_inputs, train_labels, test_inputs, test_labels


def get_svhn(data_dir: Path) -> Tuple[np.ndarray, Tensor, np.ndarray, Tensor]:
    train_dataset = SVHN(data_dir / "SVHN", split="train", download=True)
    test_dataset = SVHN(data_dir / "SVHN", split="test", download=True)

    train_inputs = train_dataset.data.transpose(0, 2, 3, 1)  # np.uint8, [73_257, 32, 32, 3]
    test_inputs = test_dataset.data.transpose(0, 2, 3, 1)  # np.uint8, [26_032, 32, 32, 3]

    train_labels = torch.tensor(train_dataset.labels)  # torch.int64, [73_257,]
    test_labels = torch.tensor(test_dataset.labels)  # torch.int64, [26_032,]

    return train_inputs, train_labels, test_inputs, test_labels


def get_data(data_dir: str, dataset: str) -> Tuple[np.ndarray, Tensor, np.ndarray, Tensor]:
    data_dir = Path(data_dir)

    if dataset == "CIFAR10":
        return get_cifar10(data_dir)

    elif dataset == "FashionMNIST":
        return get_fashionmnist(data_dir)

    elif dataset == "SVHN":
        return get_svhn(data_dir)

    else:
        raise ValueError
