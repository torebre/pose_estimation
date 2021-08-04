import torch

from pose_estimation2.load_data import load_test_data
from pose_estimation2.model_setup import get_model


def get_accuracy(model, dataloader):
    model.eval()


    for image, label in dataloader:
        # image, label = image.to('cuda'), label.to('cuda')
        output = model(image)
        prediction = torch.argmax(output)

        print("Prediction: ", prediction.item(), ". Label: ", label.item())


    model.train()



if __name__ == "__main__":
    dataloader = load_test_data()
    model = get_model()
    model.load_state_dict(torch.load("svnh_model.pt"))
    get_accuracy(model, dataloader)