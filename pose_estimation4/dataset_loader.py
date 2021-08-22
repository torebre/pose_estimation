import json



def clean_data(val_keypoint_data):
    val_keypoint_data['categories']
    keypoint_names = val_keypoint_data['categories'][0]['keypoints']

    # example_image = val_keypoint_data['annotations'][0]
    # leading_zeros = "0" * (12 - len(str(example_image['image_id'])))

    images_to_use = list(filter(filter_function, val_keypoint_data["annotations"]))

    print("Test30: ", len(images_to_use))

    # image_file = f"val2017/{leading_zeros}{example_image['image_id']}.jpg"
    # training_image = plt.imread(image_file)

    # bbox = example_image['bbox']

    # im = Image.open(image_file)

    # keypoints = example_image['keypoints']
    # drawing = ImageDraw.Draw(im)

    # for i in range(len(keypoint_names)):
    #     start = i * 3
    #     if keypoints[start + 2] == 2:
    #         Keypoint is visible
            # drawing.text((keypoints[start], keypoints[start + 1]), keypoint_names[i])

    # im = im.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])) \
    #     .resize((256, 192))


def filter_function(annotation):
    is_crowd = annotation["iscrowd"] == 1
    no_keypoint_in_image = all(keypoint == 0 for keypoint in annotation["keypoints"])
    bounding_box_too_small = annotation["bbox"][2] < 30 or annotation["bbox"][3] < 30

    return is_crowd or no_keypoint_in_image or bounding_box_too_small



if __name__ == "__main__":
    with open("../annotations/person_keypoints_val2017.json") as val_keypoints:
        val_keypoint_data = json.load(val_keypoints)

    clean_data(val_keypoint_data)
