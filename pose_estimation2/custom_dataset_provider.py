import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


class SVHN_dataset(Dataset):

    def __init__(self, data):
        images = torch.tensor(data['X']).permute([3, 2, 0, 1])
        self.labels = torch.tensor(data['y'])
        self.size = self.labels.shape[0]

        # Replace label 10 with label 0
        for label in self.labels:
            if label.item() == 10:
                label[0] = 0

        # Convert to float and normalize to [0, 1] range
        self.normalized_images = [image.type(torch.FloatTensor) / 255 for image in self.images]
        for image in self.normalized_images:
            image.type(torch.FloatTensor)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


if __name__ == "__main__":
    test_dataset = loadmat("../svhn/test_32x32.mat")
    dataset = SVHN_dataset(test_dataset)
