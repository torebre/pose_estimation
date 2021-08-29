import torch.nn as nn
from torchsummary import summary


def get_model():
    return nn.Sequential(
        nn.Conv2d(stride=2, padding=3, kernel_size=7, in_channels=3, out_channels=64),
        nn.BatchNorm2d(num_features=64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(kernel_size=5, stride=1, padding=2, out_channels=128, in_channels=64),
        nn.BatchNorm2d(num_features=128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(kernel_size=5, out_channels=256, stride=1, padding=2, in_channels=128),
        nn.BatchNorm2d(num_features=256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # nn.Flatten(),
        # nn.Linear(in_features=24 * 4 * 4, out_features=10)

        nn.Conv2d(kernel_size=1, out_channels=17, stride=1, padding=0, in_channels=256),

        nn.ConvTranspose2d(kernel_size=4, out_channels=256, stride=2, padding=1, in_channels=17),
        nn.BatchNorm2d(num_features=256),
        nn.ReLU(),

        nn.ConvTranspose2d(kernel_size=4, out_channels=17, stride=2, padding=1, in_channels=256),
        nn.BatchNorm2d(num_features=17),
        nn.ReLU(),
    )


if __name__ == "__main__":
    model = get_model()
    summary(model, (3, 256, 192))