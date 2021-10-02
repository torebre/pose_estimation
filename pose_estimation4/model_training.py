import json
import os

import torch
import torch.nn as nn
from torch import optim
from torchsummary import summary

from pose_estimation4.dataset_loader import CocoDataset
from pose_estimation4.model_setup import get_model


learning_rate = 1e-3

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

# with open("../annotations/person_keypoints_val2017.json") as val_keypoints:
#     image_directory = f"{dir_path}/../val2017"
#     val_keypoint_data = json.load(val_keypoints)
#     validation_dataset = CocoDataset(val_keypoint_data, image_directory)

# test_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=False)


for epoch in range(n_epochs):
    for images, heatmaps, validities in training_dataloader:
        # Set training to use CUDA
        images = images.to('cuda')
        heatmaps = heatmaps.to('cuda')
        validities = validities.to('cuda')

        # summary(model, (3, 256, 192))
        # print(f"Test25: {heatmaps[0].shape}, {validities[0].shape}")

        outputs = model(images)

        # TODO Setup correct use of loss function
        # print(f"Outputs: {outputs.shape}. Heatmaps: {heatmaps.shape}")

        batch_size = validities.shape[0]
        validities_expanded = validities.view(batch_size, -1, 1, 1)

        # print(f"Test30: {validities.shape}, {validities_expanded.shape}, {outputs.shape}")

        loss = loss_fn(outputs * validities_expanded, heatmaps * validities_expanded)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

    # if epoch % 10 == 0:
        # accuracy = get_accuracy(model, test_dataloader)
        # print(f"Accuracy:{accuracy}")
        #
        # if accuracy > 0.9:
        #     break

# torch.save(model.state_dict(), "svnh_model_normalized_images.pt")
