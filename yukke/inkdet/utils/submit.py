import numpy as np


def rle(img):
    """Encode predicted mask to the run-length information.

    ref: https://www.kaggle.com/stainsby/fast-tested-rle
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)
