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

# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.net = nn.Sequential(
#         nn.Conv2d(stride=1, padding=2, kernel_size=5, in_channels=3, out_channels=6),
#         nn.BatchNorm2d(num_features=6),
#         nn.ReLU(),
#         nn.MaxPool2d(kernel_size=2, stride=2),
#
#         nn.Conv2d(kernel_size=3, stride=1, padding=1, out_channels=12, in_channels=6),
#         nn.BatchNorm2d(num_features=12),
#         nn.ReLU(),
#         nn.MaxPool2d(kernel_size=2, stride=2),
#
#         nn.Conv2d(kernel_size=3, out_channels=24, stride=1, padding=1, in_channels=12),
#         nn.BatchNorm2d(num_features=24),
#         nn.ReLU(),
#         nn.MaxPool2d(kernel_size=2, stride=2),
#
#         # TODO What are the dimensions to use here?
#         nn.Linear(in_features= 16 * 16, out_features=10)
#     )

model = nn.Sequential(
    nn.Conv2d(stride=1, padding=2, kernel_size=5, in_channels=3, out_channels=6),
    nn.BatchNorm2d(num_features=6),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(kernel_size=3, stride=1, padding=1, out_channels=12, in_channels=6),
    nn.BatchNorm2d(num_features=12),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(kernel_size=3, out_channels=24, stride=1, padding=1, in_channels=12),
    nn.BatchNorm2d(num_features=24),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    # TODO What are the dimensions to use here?
    nn.Flatten(),
    nn.Linear(in_features= 24 * 4 * 4, out_features=10)
).to('cuda')

    # def forward(self, x):
    #     for layer in self.net:
    #         x = layer(x)
    #         print(x.size())
    #     return x

learning_rate = 1e-3
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# loss_fn = nn.NLLLoss()

# model = Model()

summary(model, (3, 32, 32))

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

n_epochs = 100
dataset = load_data()

# print("Model: ", model)


print("Cuda device count: ", torch.cuda.device_count())


for epoch in range(n_epochs):
    for images, labels in dataset:
        batch_size = images.shape[0]
        images, labels = images.to('cuda'), labels.to('cuda')

        # print("Batch size: ", images.shape)

        # outputs = model(images.view(batch_size - 1))
        outputs = model(images)
        # out = model(images.view(-1).unsqueeze(0))

        # print("Outputs: ", outputs)

        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))
