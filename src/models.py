from functools import partial
from typing import Tuple

from torch import Tensor
from torch.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    Dropout2d,
    Identity,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
)
from torch.nn.functional import dropout, max_pool2d, relu, sigmoid
from torchvision.models import resnet18


class LeNet5FeatureExtractor(Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = Dropout2d()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = max_pool2d(x, 2)
        x = relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = max_pool2d(x, 2)
        x = relu(x)
        x = x.reshape(-1, 320)
        return x


class VGG16FeatureExtractor(Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        # fmt: off
        layer_sizes = (
            64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"
        )
        # fmt: on

        layers = []

        for out_channels in layer_sizes:
            if out_channels == "M":
                layers += [MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    BatchNorm2d(num_features=out_channels),
                    ReLU(inplace=True),
                ]
                in_channels = out_channels

        layers += [AvgPool2d(kernel_size=1, stride=1)]

        self.layers = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        x = x.reshape(len(x), -1)
        return x


class ResNet18FeatureExtractor(Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.layers = resnet18()
        self.layers.conv1 = Conv2d(
            in_channels, out_channels=64, kernel_size=3, stride=1, padding=2, bias=False
        )
        self.layers.maxpool = Identity()
        self.layers.fc = Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x


class TaskClassifier(Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.fc1 = Linear(in_features=input_size, out_features=50)
        self.fc2 = Linear(in_features=50, out_features=10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = relu(x)
        x = dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class PoolClassifier(Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.fc1 = Linear(in_features=input_size, out_features=50)
        self.fc2 = Linear(in_features=50, out_features=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = relu(x)
        x = dropout(x, training=self.training)
        x = self.fc2(x)
        x = sigmoid(x)
        return x


def get_models(dataset: str, use_resnet: bool = False) -> Tuple[Module, Module, Module]:
    if dataset in {"CIFAR10", "SVHN"}:
        feat_model = ResNet18FeatureExtractor if use_resnet else VGG16FeatureExtractor
        task_model = partial(TaskClassifier, input_size=512)
        pool_model = partial(PoolClassifier, input_size=512)

    elif dataset == "FashionMNIST":
        assert use_resnet is False
        feat_model = LeNet5FeatureExtractor
        task_model = partial(TaskClassifier, input_size=320)
        pool_model = partial(PoolClassifier, input_size=320)

    else:
        raise ValueError

    return feat_model, task_model, pool_model
