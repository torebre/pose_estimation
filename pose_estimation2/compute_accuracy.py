import random

import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat

from pose_estimation2.custom_dataset_provider import SVHN_dataset
from pose_estimation2.model_setup import get_model


def get_accuracy(model, dataloader):
    model.eval()

    correctly_classified = 0
    number_of_images = len(dataloader)

    for image, label in dataloader:
        output = model(image.to("cuda"))
        prediction = torch.argmax(output)

        if prediction.item() == label.item():
            correctly_classified += 1

    model.train()

    return correctly_classified / number_of_images


def show_classification_examples(model, dataset):
    model.eval()

    correctly_classified = 0
    number_of_images = len(dataset)
    random.seed(1)

    while correctly_classified < 5:
        image_number = random.randrange(number_of_images)
        image, label = dataset[image_number]

        output = model(torch.unsqueeze(image, 0))
        prediction = torch.argmax(output)

        if prediction.item() == label:
            correctly_classified += 1

            plt.imshow(dataset.data['X'][:, :, :, image_number])
            plt.show()
            print(f"Digit label: {label}")

    model.train()


if __name__ == "__main__":
    # dataloader = load_test_data()

    # dirname = os.path.join(os.path.dirname(__file__), "../svhn")
    # Does this transform also divide by 255?
    # svhn_data = torchvision.datasets.SVHN(dirname, split="test", transform=
    # torchvision.transforms.ToTensor())

    test_data = loadmat("../svhn/test_32x32.mat")
    test_dataset = SVHN_dataset(test_data)

    model = get_model()
    model.load_state_dict(torch.load("svnh_model_normalized_images.pt"))
    # accuracy = get_accuracy(model, dataloader)
    # print(f"Accuracy: {accuracy}")

    show_classification_examples(model, test_dataset)
