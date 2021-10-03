import json
import os

import torch

from pose_estimation4.dataset_loader import CocoDataset


def create_tensor_files(data_file: str, image_directory: str, output_directory: str):
    with open(data_file) as val_keypoints:
        val_keypoint_data = json.load(val_keypoints)
        dataset = CocoDataset(val_keypoint_data, image_directory)

        counter = 0
        for sample in dataset:
            print(f"Data set: {sample[0].shape}, {sample[1].shape}, {sample[2].shape}")
            torch.save(sample, output_directory +"/sample_" +str(counter) +".pt")
            counter += 1


        # example_input = dataset[0]


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    image_directory = f"{dir_path}/../val2017"
    output_directory = f"{dir_path}/../tensor_files"
    create_tensor_files("../annotations/person_keypoints_val2017.json", image_directory, output_directory)