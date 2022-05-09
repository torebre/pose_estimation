import json
import os
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from pose_estimation4.heatmap_computations import generate_heatmap_for_image, apply_gaussian_filter_to_heatmaps
from pose_estimation4.load_data import load_image


class CocoDataset(Dataset):
    images_to_use: List
    image_directory: str
    mean_values = np.array([0.485, 0.456, 0.406])
    scale_values = np.array([0.229, 0.224, 0.225])

    def __init__(self, val_keypoint_data, image_directory: str):
        self.images_to_use = list(filter(filter_function, val_keypoint_data["annotations"]))
        self.image_directory = image_directory
        print(f"Number of images: {len(self.images_to_use)}")

    def __len__(self):
        return len(self.images_to_use)

    def __getitem__(self, idx):
        (image_data, transformed_pixel_array) = self.get_image(idx)
        heatmaps, validity = generate_heatmap_for_image(image_data)
        apply_gaussian_filter_to_heatmaps(heatmaps)

        image_tensor = torch.from_numpy(transformed_pixel_array.astype('float32')).permute([2, 1, 0])

        return image_tensor, torch.from_numpy(heatmaps.astype('float32')), torch.from_numpy(
            validity.astype('float32'))

    def get_image(self, idx):
        image_data = self.images_to_use[idx]
        leading_zeros = "0" * (12 - len(str(image_data['image_id'])))
        image_file = f"{self.image_directory}/{leading_zeros}{image_data['image_id']}.jpg"
        image_as_pixel_array = load_image(image_file, image_data)

        # transformed_pixel_array = image_as_pixel_array.astype('float32')

        # Scale the image data using the mean and variance given in the exercise text
        # transformed_pixel_array[:, :, 0] = (transformed_pixel_array[:, :, 0] - self.mean_values[0]) / self.scale_values[
        #     0]
        # transformed_pixel_array[:, :, 1] = (transformed_pixel_array[:, :, 1] - self.mean_values[1]) / self.scale_values[
        #     0]
        # transformed_pixel_array[:, :, 2] = (transformed_pixel_array[:, :, 2] - self.mean_values[2]) / self.scale_values[
        #     0]

        mean = np.asarray([0.485, 0.456, 0.406])
        std = np.asarray([0.229, 0.224, 0.225])
        transformed_pixel_array = image_as_pixel_array.astype('float32') / 255.0
        transformed_pixel_array = (transformed_pixel_array - mean) / std

        # return image_data, image_as_pixel_array
        return image_data, transformed_pixel_array


def clean_data(val_keypoint_data) -> List:
    images_to_use = list(filter(filter_function, val_keypoint_data["annotations"]))
    return images_to_use


def filter_function(annotation):
    is_crowd = annotation["iscrowd"] == 1
    no_keypoint_in_image = all(keypoint == 0 for keypoint in annotation["keypoints"])
    bounding_box_too_small = annotation["bbox"][2] < 30 or annotation["bbox"][3] < 30

    return not (is_crowd or no_keypoint_in_image or bounding_box_too_small)


if __name__ == "__main__":
    with open("../annotations/person_keypoints_val2017.json") as val_keypoints:
        val_keypoint_data = json.load(val_keypoints)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        image_directory = f"{dir_path}/../val2017"
        dataset = CocoDataset(val_keypoint_data, image_directory)

        # for sample in dataset:
        #     print(f"Data set: {sample[0].shape}, {sample[1].shape}, {sample[2].shape}")

        # image_metadata, sample_image = dataset.get_image(idx=1)
        # plt.imshow(sample_image)
        # plt.show()

        image_tensor, heatmap, validity = dataset[0]
        image_tensor_np = image_tensor.cpu().numpy()
        heatmap_np = heatmap.cpu().numpy()

        print("Image: ", image_tensor_np.shape, image_tensor_np.dtype, np.max(image_tensor_np), np.min(image_tensor_np))
        print("Heatmap: ", heatmap_np.shape, heatmap_np.dtype, np.max(heatmap_np), np.min(heatmap_np))
        print("Validity: ", validity.shape, validity.dtype)

        image_tensor_np = image_tensor_np.transpose(1, 2, 0)
        mean = np.asarray([0.485, 0.456, 0.406])
        std = np.asarray([0.229, 0.224, 0.225])
        image_tensor_np = image_tensor_np * std + mean

        figure = plt.figure(2, figsize=(20, 20))
        sub1 = figure.add_subplot(121)
        sub1.imshow(image_tensor_np)
        sub2 = figure.add_subplot(122)
        sub2.imshow(np.sum(heatmap_np, axis=0))
        plt.show()

        # figure2 = plt.figure(20)
        # for i in range(0, heatmap_np.shape[0]):
        #     subplot = figure2.add_subplot(4, 5, i + 1)
        #     subplot.imshow(heatmap_np[i])
        # plt.show()

        # figure2 = plt.figure()
        # plt.imshow(np.sum(heatmap_np, axis=0))
        # plt.show()
