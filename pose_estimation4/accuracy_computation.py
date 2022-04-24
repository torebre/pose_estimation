import random

import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat

from pose_estimation2.custom_dataset_provider import SVHN_dataset
from pose_estimation2.model_setup import get_model
from pose_estimation4.pck_metric import accuracy


def get_accuracy(model, dataloader: torch.utils.data.DataLoader):
    model.eval()

    # Only use the first batch for validation to speed up the process
    sample = next(iter(dataloader))
    images = sample[0].to('cuda')
    heatmaps = sample[1]

    heatmaps_as_array = heatmaps.detach().cpu().numpy()

    outputs = model(images)

    output_as_array = outputs.detach().cpu().numpy()
    computed_accuracy = accuracy(output_as_array, heatmaps_as_array)

    model.train()

    return computed_accuracy
