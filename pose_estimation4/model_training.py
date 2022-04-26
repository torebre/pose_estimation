import json
import os

import torch
import torch.nn as nn
from torch import optim

from pose_estimation4.accuracy_computation import get_accuracy
from pose_estimation4.dataset_loader import CocoDataset


def train_model(model, learning_rate=1e-4, learning_rate_updated=1e-5,
                number_of_epochs=10, drive_path="drive/human_pose_estimation/section4"):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [8], 0.1)
    loss_fn = nn.MSELoss()

    dir_path = os.path.dirname(os.path.realpath(__file__))

    def checkpoint_model(model, optimizer, epoch, iteration_counter):
        # torch.save(model.state_dict(), DRIVE_PATH + "network_" + str(epoch) + "_" + str(batch) + '.pt')
        torch.save(model.state_dict(), "network_" + str(epoch) + "_" + str(iteration_counter) + '.pt')

    with open("../annotations/person_keypoints_train2017.json") as train_keypoints:
        image_directory = f"{dir_path}/../train2017"
        train_keypoint_data = json.load(train_keypoints)
        training_dataset = CocoDataset(train_keypoint_data, image_directory)

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=32, shuffle=True)

    with open("../annotations/person_keypoints_val2017.json") as val_keypoints:
        image_directory = f"{dir_path}/../val2017"
        val_keypoint_data = json.load(val_keypoints)
        validation_dataset = CocoDataset(val_keypoint_data, image_directory)

    test_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=False)

    learning_rate_latch = True
    previous_accuracy = 0.0

    iteration_counter = 0
    for epoch in range(number_of_epochs):
        # Iterate over the batches
        for images, heatmaps, validities in training_dataloader:
            # Set training to use CUDA
            images = images.to('cuda')
            heatmaps = heatmaps.to('cuda')
            validities = validities.to('cuda')

            outputs = model(images)

            batch_size = validities.shape[0]
            validities_expanded = validities.view(batch_size, -1, 1, 1)
            loss = loss_fn(outputs * validities_expanded, heatmaps * validities_expanded)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iteration_counter += 1

            if iteration_counter % 100 == 0:
                computed_accuracy = get_accuracy(model, test_dataloader)
                print(f"Epoch: {epoch}. Counter: {iteration_counter}. Computed accuracy: {computed_accuracy}")

            if iteration_counter % 100 == 0:
                checkpoint_model(model, optimizer, epoch, iteration_counter)

            # scheduler.step()

        computed_accuracy = get_accuracy(model, test_dataloader)

        if learning_rate_latch and computed_accuracy <= previous_accuracy:
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate_updated
                learning_rate_latch = False

        print(f"Accuracy:{computed_accuracy}")

        if computed_accuracy > 0.9:
            break
