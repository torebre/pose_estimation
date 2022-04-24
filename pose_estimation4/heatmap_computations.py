import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter


def generate_heatmaps(val_keypoint_data):
    val_keypoint_data["annotations"]

    for image in val_keypoint_data["annotations"]:
        generate_heatmap_for_image(image)


def generate_heatmap_for_image(image) -> tuple:
    heatmaps = np.zeros((17, 64, 48))
    keypoints = image["keypoints"]
    bbox = image['bbox']
    width = bbox[2]
    height = bbox[3]
    validity = np.zeros(17)

    for i in range(17):
        start = i * 3
        visibility_flag = keypoints[start + 2]

        if visibility_flag != 0:
            # Keypoint is in the image, it may be hidden (1) or visible (2)
            x_coordinate = keypoints[start]
            y_coordinate = keypoints[start + 1]

            x_scaling_factor = 64.0 * (x_coordinate - bbox[0]) / width
            y_scaling_factor = 48.0 * (y_coordinate - bbox[1]) / height

            if x_scaling_factor > 1.0 \
                    or x_scaling_factor < 0.0 \
                    or y_scaling_factor > 1.0 \
                    or y_scaling_factor < 0.0:
                # This means the feature is not inside the bounding box, skip
                # the image in that case
                continue

            x_scaled = np.floor(x_scaling_factor)
            y_scaled = np.floor(y_scaling_factor)

            heatmaps[i, x_scaled.astype(int), y_scaled.astype(int)] = 1
            validity[i] = 1

    return heatmaps, validity


def apply_gaussian_filter_to_heatmaps(heatmaps):
    for i in range(heatmaps.shape[0]):
        heatmaps[i, :, :] = gaussian_filter(heatmaps[i, :, :], sigma=2)
        max_value = np.max(heatmaps[i, :, :])

        # If the max value is 0 it means that the key point is
        # not visible in the image
        if max_value > 0.0:
            heatmaps[i, :, :] /= max_value


def generate_heatmap_test(val_keypoint_data):
    keypoint_names = val_keypoint_data['categories'][0]['keypoints']
    heatmaps = generate_heatmap_for_image(val_keypoint_data["annotations"][0])
    apply_gaussian_filter_to_heatmaps(heatmaps)

    # print(heatmaps[6, :, :])

    plt.imshow(heatmaps[6, :, :])
    plt.show()


if __name__ == "__main__":
    with open("../annotations/person_keypoints_val2017.json") as val_keypoints:
        val_keypoint_data = json.load(val_keypoints)

    # val_keypoint_data['categories']
    # keypoint_names = val_keypoint_data['categories'][0]['keypoints']

    # example_image_annotations = val_keypoint_data['annotations'][0]
    # leading_zeros = "0" * (12 - len(str(example_image_annotations['image_id'])))

    # image_file = f"../val2017/{leading_zeros}{example_image_annotations['image_id']}.jpg"
    # training_image = plt.imread(image_file)

    # generate_heatmaps(val_keypoint_data)

    generate_heatmap_test(val_keypoint_data)
