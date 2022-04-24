import json
import os

import torch
import torch.nn as nn
from torch import optim

from pose_estimation4.accuracy_computation import get_accuracy
from pose_estimation4.dataset_loader import CocoDataset
from pose_estimation4.model_setup import get_model

LEARNING_RATE = 1e-4
LEARNING_RATE_UPDATED = 1e-5
NUMBER_OF_EPOCHS = 10
DRIVE_PATH = "drive/human_pose_estimation/section4"

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

model = get_model().to('cuda')

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
for epoch in range(NUMBER_OF_EPOCHS):
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
            param_group["lr"] = LEARNING_RATE_UPDATED
            learning_rate_latch = False

    print(f"Accuracy:{computed_accuracy}")

    if computed_accuracy > 0.9:
        break
