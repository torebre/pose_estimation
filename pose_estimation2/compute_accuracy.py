import torch

from pose_estimation2.load_data import load_test_data
from pose_estimation2.model_setup import get_model


def get_accuracy(model, dataloader):
    model.eval()

    correctly_classified = 0
    number_of_images = len(dataloader)

    for image, label in dataloader:
        # image, label = image.to('cuda'), label.to('cuda')
        output = model(image)
        prediction = torch.argmax(output)

        # print("Prediction: ", prediction.item(), ". Label: ", label.item())

        if prediction.item() == label.item():
            correctly_classified += 1

    model.train()

    return correctly_classified / number_of_images


if __name__ == "__main__":
    dataloader = load_test_data()
    model = get_model()
    model.load_state_dict(torch.load("svnh_model.pt"))
    accuracy = get_accuracy(model, dataloader)

    print(f"Accuracy: {accuracy}")
