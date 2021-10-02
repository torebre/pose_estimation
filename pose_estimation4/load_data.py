import json

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw


def load_image(image_file, image_annotation) -> npt.ArrayLike:
    bbox = image_annotation['bbox']
    im = Image.open(image_file).convert('RGB')

    keypoints = image_annotation['keypoints']
    drawing = ImageDraw.Draw(im)

    im = im.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])) \
        .resize((256, 192))

    pixel_array = np.array(im) / 255.0

    # print(pixel_array)

    # print("Test23: ", pixel_array.shape)
    # print("Test24: ", np.mean(pixel_array[:, :, 1]))

    # TODO Need to normalize with respect to some mean

    return pixel_array



if __name__ == "__main__":
    with open("../annotations/person_keypoints_val2017.json") as val_keypoints:
        val_keypoint_data = json.load(val_keypoints)

    # val_keypoint_data['categories']
    # keypoint_names = val_keypoint_data['categories'][0]['keypoints']

    example_image_annotations = val_keypoint_data['annotations'][0]
    leading_zeros = "0" * (12 - len(str(example_image_annotations['image_id'])))

    image_file = f"../val2017/{leading_zeros}{example_image_annotations['image_id']}.jpg"
    # training_image = plt.imread(image_file)

    load_image(image_file, example_image_annotations)
