import json
from typing import List

import torchvision.models.detection
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)


def test_classification():
    with open("../annotations/instances_val2017.json") as annotations:
        annotations_data = json.load(annotations)

    count = 0
    for image_annotation in annotations_data["annotations"]:
        category_id = image_annotation["category_id"]

        # Category 1 is person
        if category_id != 1:
            continue

        image_id = image_annotation["image_id"]
        image = load_image(image_id)
        person_box_images = return_cropped_and_resized_images(image)

        for _, box_image in person_box_images:
            box_image.show()

        count += 1
        if count == 5:
            break


def return_cropped_and_resized_images(image) -> List[tuple[List[float], Image.Image]]:
    image_as_tensor = transform_image(image)
    classification = classify_image(image_as_tensor)
    valid_prediction_boxes = get_predictions_above_threshold(classification)

    return [(box, image.crop(box).resize((256, 192))) for box in valid_prediction_boxes]


def draw_boxes_on_image(classification, image):
    drawing = ImageDraw.Draw(image)

    for prediction in classification:
        count = -1
        for label in prediction["labels"]:
            count += 1
            if label.item() != 1:
                continue

            if prediction["scores"][count].item() < 0.9:
                continue

            drawing.rectangle(prediction["boxes"][count].tolist())

            print("Test23: ", prediction["boxes"][count].tolist()[:2])
            drawing.text(prediction["boxes"][count].tolist()[:2], str(prediction["scores"][count].item()), fill="red")

    image.show()


def get_predictions_above_threshold(classification):
    predictions = []

    for prediction in classification:
        count = -1
        for label in prediction["labels"]:
            count += 1
            if label.item() != 1:
                continue

            if prediction["scores"][count].item() < 0.9:
                continue

            predictions.append(prediction["boxes"][count].tolist())

    return predictions



def transform_image(image):
    # This transformation scales the pixel values to be in the [0, 1] range
    transform = torchvision.transforms.ToTensor()
    return transform(image)


def load_image(image_id) -> Image:
    leading_zeros = "0" * (12 - len(str(image_id)))
    image_file = f"../val2017/{leading_zeros}{image_id}.jpg"
    training_image = plt.imread(image_file)

    return Image.open(image_file)


def classify_image(image_as_tensor):
    model.eval()
    output = model([image_as_tensor])
    model.train()
    return output


test_classification()


# {
#     "supercategory": "person",
#     "id": 1,
#     "name": "person"
# }

