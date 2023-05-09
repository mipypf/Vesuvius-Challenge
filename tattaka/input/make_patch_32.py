import glob
import os
import warnings

import numpy as np
import PIL.Image as Image
from tqdm import tqdm, trange

warnings.simplefilter("ignore")

PREFIX = "vesuvius-challenge-ink-detection/train"
PATCH_SIZE = 32

data_ids = [1, 2, 3]

for data_id in tqdm(data_ids):
    ir = np.array(Image.open(PREFIX + f"/{data_id}/ir.png"))
    mask = np.array(Image.open(PREFIX + f"/{data_id}/mask.png").convert("1"))
    label = np.array(Image.open(PREFIX + f"/{data_id}/inklabels.png"))
    volume = np.stack(
        [
            np.array(Image.open(filename), dtype=np.float32) / 65535.0
            for filename in sorted(
                glob.glob(PREFIX + f"/{data_id}/surface_volume/*.tif")
            )
        ]
    )
    assert ir.shape[-2:] == mask.shape[-2:] == volume.shape[-2:]
    volume_dir = f"vesuvius_patches_{PATCH_SIZE}/train/{data_id}/surface_volume/"
    ir_dir = f"vesuvius_patches_{PATCH_SIZE}/train/{data_id}/ir/"
    mask_dir = f"vesuvius_patches_{PATCH_SIZE}/train/{data_id}/mask/"
    label_dir = f"vesuvius_patches_{PATCH_SIZE}/train/{data_id}/label/"
    os.makedirs(volume_dir, exist_ok=True)
    os.makedirs(ir_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    h, w = volume.shape[-2:]
    for i in trange(h // PATCH_SIZE, leave=False):
        for j in trange(w // PATCH_SIZE, leave=False):
            start_h = i * PATCH_SIZE
            start_w = j * PATCH_SIZE
            mask_patch = mask[
                ..., start_h : start_h + PATCH_SIZE, start_w : start_w + PATCH_SIZE
            ]
            if not mask_patch.sum():
                continue
            volume_patch = volume[
                ..., start_h : start_h + PATCH_SIZE, start_w : start_w + PATCH_SIZE
            ]
            ir_patch = ir[
                ..., start_h : start_h + PATCH_SIZE, start_w : start_w + PATCH_SIZE
            ]
            label_patch = label[
                ..., start_h : start_h + PATCH_SIZE, start_w : start_w + PATCH_SIZE
            ]
            np.save(os.path.join(volume_dir, f"volume_{i}_{j}"), volume_patch)
            np.save(os.path.join(ir_dir, f"ir_{i}_{j}"), ir_patch)
            np.save(os.path.join(label_dir, f"label_{i}_{j}"), label_patch)
            np.save(os.path.join(mask_dir, f"mask_{i}_{j}"), mask_patch)
