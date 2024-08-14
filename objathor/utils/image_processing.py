from typing import Sequence

import numpy as np
from PIL import Image


def verify_images_are_not_all_white(image_paths: Sequence[str]):
    # Verify that images are not nearly completely white
    for p in image_paths:
        img = np.array(Image.open(p))
        if img.shape[-1] == 4:
            img[img[:, :, 3] == 0] = 255
            img = img[:, :, :3]

        if img.min() > 245:
            return False

    return True
