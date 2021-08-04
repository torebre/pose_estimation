import os

import torch
import torchvision


def load_data():
    dirname = os.path.join(os.path.dirname(__file__), "../svhn")
    # Does this transform also divide by 255?
    svhn_data = torchvision.datasets.SVHN(dirname, split="train", transform=
    torchvision.transforms.ToTensor())
    return torch.utils.data.DataLoader(svhn_data, batch_size=512, shuffle=True)


def load_test_data():
    dirname = os.path.join(os.path.dirname(__file__), "../svhn")
    # Does this transform also divide by 255?
    svhn_data = torchvision.datasets.SVHN(dirname, split="test", transform=
    torchvision.transforms.ToTensor())
    return torch.utils.data.DataLoader(svhn_data)
