import json
from typing import List

import os

import torch
from torch.utils.data import Dataset

from pose_estimation4.heatmap_computations import generate_heatmap_for_image, apply_gaussian_filter_to_heatmaps
from pose_estimation4.load_data import load_image


class CocoDataset(Dataset):
    images_to_use: List
    image_directory: str

    def __init__(self, val_keypoint_data, image_directory: str):
        self.images_to_use = list(filter(filter_function, val_keypoint_data["annotations"]))
        self.image_directory = image_directory
        # keypoint_names = val_keypoint_data['categories'][0]['keypoints']

    def __len__(self):
        return len(self.images_to_use)

    def __getitem__(self, idx):
        image_data = self.images_to_use[idx]
        leading_zeros = "0" * (12 - len(str(image_data['image_id'])))
        image_file = f"{self.image_directory}/{leading_zeros}{image_data['image_id']}.jpg"

        image_as_pixel_array = load_image(image_file, image_data)

        heatmaps, validity = generate_heatmap_for_image(image_data)
        apply_gaussian_filter_to_heatmaps(heatmaps)

        return torch.from_numpy(image_as_pixel_array), torch.from_numpy(heatmaps), torch.from_numpy(validity)


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
        example_input = dataset[0]

        print("Data set: ", example_input[0].shape, example_input[1].shape, example_input[2].shape)
