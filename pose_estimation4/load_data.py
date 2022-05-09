import numpy as np
import numpy.typing as npt
from PIL import Image


def load_image(image_file, image_annotation) -> npt.ArrayLike:
    bbox = image_annotation['bbox']
    im = Image.open(image_file).convert('RGB')

    im = im.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])) \
        .resize((256, 192))

    pixel_array = np.array(im) / 255.0

    return pixel_array
