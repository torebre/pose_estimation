import torch

from pose_estimation4.model_training import train_model
from pose_estimation4.resnet import resnet_pose

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

model = resnet_pose().to('cuda')

train_model(model)
