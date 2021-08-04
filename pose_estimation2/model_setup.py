import torch.nn as nn


def get_model():
    return nn.Sequential(
        nn.Conv2d(stride=1, padding=2, kernel_size=5, in_channels=3, out_channels=6),
        nn.BatchNorm2d(num_features=6),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(kernel_size=3, stride=1, padding=1, out_channels=12, in_channels=6),
        nn.BatchNorm2d(num_features=12),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(kernel_size=3, out_channels=24, stride=1, padding=1, in_channels=12),
        nn.BatchNorm2d(num_features=24),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Flatten(),
        nn.Linear(in_features=24 * 4 * 4, out_features=10)
    )
