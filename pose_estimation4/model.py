import torch

from pose_estimation4.model_training import train_model
from pose_estimation4.model_setup import get_model


device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

model = get_model().to('cuda')

train_model(model)
