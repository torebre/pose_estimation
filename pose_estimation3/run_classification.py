import json

import torch
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

        print(f"Category ID: {category_id}")

        image_as_tensor = transform_image(image)
        classification = classify_image(image_as_tensor)

        draw_boxes_on_image(classification, image)

        count += 1
        if count == 5:
            break



    # keypoints = example_image['keypoints']
    # drawing = ImageDraw.Draw(im)
    #
    # for i in range(len(keypoint_names)):
    #     start = i * 3
    #     if keypoints[start + 2] == 2:
            Keypoint is visible
            # drawing.text((keypoints[start], keypoints[start + 1]), keypoint_names[i])
    #
    # im = im.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])) \
    #     .resize((256, 192))


def draw_boxes_on_image(classification, image):
    drawing = ImageDraw.Draw(image)

    for prediction in classification:
        count = -1
        for label in prediction["labels"]:
            count += 1
            if label.item() != 1:
                continue

            if prediction["scores"][count].item() < 0.7:
                continue

            drawing.rectangle(prediction["boxes"][count].tolist())

    image.show()



def transform_image(image):
    # This transformation scales the pixel values to be in the [0, 1] range
    transform = torchvision.transforms.ToTensor()
    return transform(image)


def load_image(image_id):
    leading_zeros = "0" * (12 - len(str(image_id)))
    image_file = f"../val2017/{leading_zeros}{image_id}.jpg"
    training_image = plt.imread(image_file)

    return Image.open(image_file)


def classify_image(image_as_tensor):
    model.eval()
    # output = model(torch.unsqueeze(image_as_tensor, 0))
    output = model([image_as_tensor])
    model.train()

    print("Output:", output[0]["scores"])

    return output


test_classification()


# {
#     "supercategory": "person",
#     "id": 1,
#     "name": "person"
# }

