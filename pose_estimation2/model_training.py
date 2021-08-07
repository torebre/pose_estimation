import torch
import torch.nn as nn
from torchsummary import summary

# 1. convolution, kernel_siz e =5, channel s =6, strid e =1, paddin g =2
# 2. batch -normalization
# 3. ReLU
# 4. Ma x -pool, kernel_siz e =2, strid e =2
#
# 5. convolution, kernel_siz e =3, channel s =12, strid e =1, paddin g =1
# 6. batc h -normalization
# 7. ReLU
# 8. Ma x -pool, kernel_siz e =2, strid e =2
#
# 9. convolution, kernel_siz e =3, channel s =24, strid e =1, paddin g =1
# 10. batc h -normalization
# 11. ReLU
# 12. Ma x -pool, kernel_siz e =2, strid e =2
#
# 13. fully connected layer, output_siz e =10
from torch import optim

from pose_estimation2.load_data import load_data
from pose_estimation2.model_setup import get_model

model = get_model().to('cuda')

learning_rate = 1e-3

# summary(model, (3, 32, 32))

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

n_epochs = 100
dataset = load_data()

print("Cuda device count: ", torch.cuda.device_count())

for epoch in range(n_epochs):
    for images, labels in dataset:
        batch_size = images.shape[0]
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)

        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

torch.save(model.state_dict(), "svnh_model_normalized_images.pt")
