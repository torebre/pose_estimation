import torch
import torch.nn as nn
from scipy.io import loadmat

from torch import optim

from pose_estimation2.compute_accuracy import get_accuracy
from pose_estimation2.custom_dataset_provider import SVHN_dataset
from pose_estimation2.model_setup import get_model

model = get_model().to('cuda')

learning_rate = 1e-3

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

n_epochs = 100

training_data = loadmat("../svhn/train_32x32.mat")
training_dataset = SVHN_dataset(training_data)
training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=512, shuffle=True)

test_data = loadmat("../svhn/test_32x32.mat")
test_dataset = SVHN_dataset(test_data)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

for epoch in range(n_epochs):
    for images, labels in training_dataloader:
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)

        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

    if epoch % 10 == 0:
        accuracy = get_accuracy(model, test_dataloader)
        print(f"Accuracy:{accuracy}")

        if accuracy > 0.9:
            break

torch.save(model.state_dict(), "svnh_model_normalized_images.pt")
