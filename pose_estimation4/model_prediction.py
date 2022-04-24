import json
import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from pose_estimation4.dataset_loader import CocoDataset
from pose_estimation4.model_setup import get_model

dir_path = os.path.dirname(os.path.realpath(__file__))


def plot_image(sample_image, heatmap_image):
    img = sample_image.cpu().numpy()
    heatmap = heatmap_image.cpu().numpy()

    img = img[0].squeeze().transpose(1, 2, 0)
    mean = np.asarray([0.485, 0.456, 0.406])
    std = np.asarray([0.229, 0.224, 0.225])
    img = img * std + mean
    heatmap = np.sum(heatmap[0], axis=0)

    fig = plt.figure(2, figsize=(20, 20))
    plt.gray()
    ax1 = fig.add_subplot(121)
    ax1.imshow(img)
    ax2 = fig.add_subplot(122)
    ax2.imshow(heatmap)
    plt.show()


def predict_pose(pose_model, validation_data):
    sample = next(iter(validation_data))
    # Need to add an empty dimension because the model input
    # operates on a batch of images, and here there is only
    # a single image
    sample_image = sample[0][None, ...].to('cuda')
    sample_heatmap = sample[1][None, ...].to('cuda')

    pose_model.eval()
    output = pose_model(sample_image)

    plot_image(sample_image, sample_heatmap)
    plot_image(sample_image, output)


if __name__ == "__main__":
    model = get_model().to('cuda')
    model.load_state_dict(torch.load("network_0_700.pt"))

    image_directory = f"{dir_path}/../val2017"
    with open("../annotations/person_keypoints_val2017.json") as val_keypoints:
        val_keypoint_data = json.load(val_keypoints)

    validation_dataset = CocoDataset(val_keypoint_data, image_directory)

    # Need to specify no_grad here to avoid an exception when trying to use the heatmap image
    with torch.no_grad():
        predict_pose(model, validation_dataset)
