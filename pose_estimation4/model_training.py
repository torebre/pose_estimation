import json
import os

import torch
import torch.nn as nn
from torch import optim

from pose_estimation4.accuracy_computation import get_accuracy
from pose_estimation4.dataset_loader import CocoDataset
from pose_estimation4.model_setup import get_model

learning_rate = 1e-4
learning_rate_updated = 1e-5

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

model = get_model().to('cuda')
# model = get_model()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

n_epochs = 10000

dir_path = os.path.dirname(os.path.realpath(__file__))

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

for epoch in range(n_epochs):
    batch_counter = 0

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

        # TODO Using a small number of batches while developing
        batch_counter += 1
        # if counter == 1:
        #     break

        if batch_counter % 20 == 0:
            computed_accuracy = get_accuracy(model, test_dataloader)
            print(f"Counter: {batch_counter}. Computed accuracy: {computed_accuracy}")

        # print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

    computed_accuracy = get_accuracy(model, test_dataloader)

    if learning_rate_latch and computed_accuracy <= previous_accuracy:
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate_updated
            learning_rate_latch = False

    print(f"Accuracy:{computed_accuracy}")

    if computed_accuracy > 0.9:
        break

# torch.save(model.state_dict(), "svnh_model_normalized_images.pt")
