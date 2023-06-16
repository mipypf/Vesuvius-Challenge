import glob
import os
import warnings

import numpy as np
import PIL.Image as Image
from tqdm import trange

warnings.simplefilter("ignore")

PREFIX = "vesuvius-challenge-ink-detection/train"

data_id = 2
ir = Image.open(PREFIX + f"/{data_id}/ir.png")  # (h, w)
mask = Image.open(PREFIX + f"/{data_id}/mask.png")
label = Image.open(PREFIX + f"/{data_id}/inklabels.png")
volume = [
    Image.open(filename)
    for filename in sorted(glob.glob(PREFIX + f"/{data_id}/surface_volume/*.tif"))
]
mask_numpy = np.asarray(mask.convert("1"))
s1 = 0
s2 = 0
pix_sum = mask_numpy.sum()
for i in trange(mask_numpy.shape[0]):
    s1 += 1
    if mask_numpy[:s1].sum() > pix_sum // 3:
        break
s2 = s1
for i in trange(mask_numpy.shape[0] - s1):
    s2 += 1
    if mask_numpy[s1:s2].sum() > pix_sum // 3:
        break
new_data_ids = [2, 4, 5]
splits = [s1, s2, mask_numpy.shape[0]]
print(splits)
width = mask_numpy.shape[1]
NEW_PREFIX = "vesuvius-challenge-ink-detection-5fold/train"
start = 0
for i, data_id in enumerate(new_data_ids):
    os.makedirs(NEW_PREFIX + f"/{data_id}/", exist_ok=True)
    os.makedirs(NEW_PREFIX + f"/{data_id}/surface_volume/", exist_ok=True)
    ir.crop((0, start, width, splits[i])).save(NEW_PREFIX + f"/{data_id}/ir.png")
    mask.crop((0, start, width, splits[i])).save(NEW_PREFIX + f"/{data_id}/mask.png")
    label.crop((0, start, width, splits[i])).save(
        NEW_PREFIX + f"/{data_id}/inklabels.png"
    )
    for j, img in enumerate(volume):
        img.crop((0, start, width, splits[i])).save(
            NEW_PREFIX + f"/{data_id}/surface_volume/" + "{:02}.tif".format(j)
        )
    start = splits[i]
