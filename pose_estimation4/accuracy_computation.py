import random

import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat

from pose_estimation2.custom_dataset_provider import SVHN_dataset
from pose_estimation2.model_setup import get_model
from pose_estimation4.pck_metric import accuracy


def get_accuracy(model, dataloader: torch.utils.data.DataLoader):
    model.eval()

    # Only use the first batch for validation to speed up the process
    iterator = iter(dataloader)
    sample = next(iterator)

    images = sample[0].to('cuda')

    heatmaps = sample[1]
    # validities = sample[2].to('cuda')

    heatmaps_as_array = heatmaps.detach().cpu().numpy()

    # TODO Do the validities need to be taken into account here?
    outputs = model(images)

    output_as_array = outputs.detach().cpu().numpy()
    computed_accuracy = accuracy(output_as_array, heatmaps_as_array)

    model.train()

    return computed_accuracy


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
