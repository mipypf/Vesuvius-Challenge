import dataclasses
import os
from functools import partial
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from inkdet.datasets.surface_volume_dataset import SurfaceVolumeDataset
from inkdet.utils import Config, worker_init_fn
from inkdet.utils.constants import FRAGMENTS


@dataclasses.dataclass()
class Padding:
    x0: int
    y0: int
    x1: int
    y1: int

    def pad(self, image: np.ndarray, value: int) -> np.ndarray:
        image = np.pad(
            image,
            [(self.y0, self.y1), (self.x0, self.x1)],
            constant_values=value,
        )
        return image

    def slice(self, image: np.ndarray) -> np.ndarray:
        image_shape = image.shape
        return image[self.y0 : image_shape[0] - self.y1, self.x0 : image_shape[1] - self.x1]


def read_image(
    data_root: str,
    rel_filepath: str,
    patch_size: int,
    target_size: int,
    padding: Optional[Padding] = None,
    mode: int = cv2.IMREAD_GRAYSCALE,
):
    image_path = os.path.join(data_root, rel_filepath)
    assert os.path.exists(image_path)
    assert patch_size > 0
    assert target_size >= 0
    assert patch_size >= target_size

    image: np.ndarray = cv2.imread(os.path.join(data_root, rel_filepath), mode)

    if padding is None:
        pad_x0 = pad_y0 = (patch_size - target_size) // 2 if target_size > 0 else 0
        pad_y1 = patch_size - (image.shape[0] + pad_y0) % patch_size
        pad_x1 = patch_size - (image.shape[1] + pad_x0) % patch_size
        padding = Padding(x0=pad_x0, y0=pad_y0, x1=pad_x1, y1=pad_y1)

    image = padding.pad(image, value=0)

    return image, padding


def read_image_mask_label(
    data_dir: str,
    fragment_id: str,
    patch_size: int,
    cfg: Config,
    pseudo_label_fold_id: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    assert data_dir in ["train", "test"]
    assert cfg.start_channels > 0
    assert cfg.in_channels > 0
    assert cfg.in_channels + cfg.in_channels <= 65

    mask, padding = read_image(
        data_root=cfg.data_root,
        rel_filepath=f"{data_dir}/{fragment_id}/mask.png",
        patch_size=patch_size,
        target_size=cfg.target_size,
    )
    mask = mask.astype("float32")
    mask /= 255.0

    start = cfg.start_channels
    end = cfg.start_channels + cfg.in_channels
    images = []
    for i in tqdm(range(start, end)):
        image, _ = read_image(
            data_root=cfg.data_root,
            rel_filepath=f"{data_dir}/{fragment_id}/surface_volume/{i:02}.tif",
            patch_size=patch_size,
            target_size=cfg.target_size,
            padding=padding,
        )
        images.append(image)

    images = np.stack(images, axis=2)

    if data_dir == "train":
        label, _ = read_image(
            data_root=cfg.data_root,
            rel_filepath=f"train/{fragment_id}/inklabels.png",
            patch_size=patch_size,
            target_size=cfg.target_size,
            padding=padding,
        )
        label = label.astype("float32")
        label /= 255.0

        if cfg.use_pseudo_label and pseudo_label_fold_id is not None:
            pseudo_label_path = f"train/{fragment_id}/pseudo_inklabels_fold{pseudo_label_fold_id}.png"
            pseudo_label, _ = read_image(
                data_root=cfg.data_root,
                rel_filepath=pseudo_label_path,
                patch_size=patch_size,
                target_size=cfg.target_size,
                padding=padding,
                mode=cv2.IMREAD_UNCHANGED,
            )
            pseudo_label = pseudo_label.astype("float32")
            pseudo_label /= np.iinfo(np.uint16).max
            logger.info(f"read pseudo label (fold {pseudo_label_fold_id}): {pseudo_label_path}")
        else:
            pseudo_label = None
    else:
        label = None
        pseudo_label = None

    return images, mask, label, pseudo_label


def load_patched_images(
    data_dir: str,
    fragment_ids: List[str],
    patch_size: int,
    stride: int,
    cfg: Config,
    pseudo_label_fold_id: Optional[str] = None,
):
    images = []
    masks = []
    labels = []
    pseudo_labels = []
    xyxys = []
    ids = []

    assert patch_size > 0
    assert stride > 0
    assert len(cfg.clip) == 0 or len(cfg.clip) == 2
    assert cfg.gamma is None or (cfg.gamma is not None and cfg.gamma > 0)

    logger.info(f"fragment_ids: {fragment_ids}, patch_size={patch_size} stride={stride}")

    for fragment_id in fragment_ids:
        image, mask, label, pseudo_label = read_image_mask_label(
            data_dir,
            fragment_id,
            patch_size,
            cfg,
            pseudo_label_fold_id,
        )
        # image = np.rot90(image).copy()
        # label = np.rot90(label).copy()
        # mask = np.rot90(mask).copy()

        if len(cfg.clip) > 0:
            logger.info(f"Clipping: [{cfg.clip[0]}, {cfg.clip[1]}]")
            np.clip(image, a_min=cfg.clip[0], a_max=cfg.clip[1], out=image)

        if cfg.gamma is not None:
            logger.info(f"Gamma Correction: {cfg.gamma}")
            dtype = image.dtype
            image_f = image.astype(np.float32) / 255.0
            image = (255.0 * image_f ** (1 / cfg.gamma)).astype(dtype)

        x1_list = list(range(0, image.shape[1] - patch_size + 1, stride))
        y1_list = list(range(0, image.shape[0] - patch_size + 1, stride))

        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + patch_size
                x2 = x1 + patch_size

                images.append(image[y1:y2, x1:x2])
                masks.append(mask[y1:y2, x1:x2])
                xyxys.append([x1, y1, x2, y2])
                ids.append(fragment_id)

                if label is not None:
                    labels.append(label[y1:y2, x1:x2, None])
                if pseudo_label is not None:
                    pseudo_labels.append(pseudo_label[y1:y2, x1:x2, None])

    labels = labels if len(labels) > 0 else None
    pseudo_labels = pseudo_labels if len(pseudo_labels) > 0 else None

    return images, masks, labels, pseudo_labels, xyxys, ids


def get_transforms(split: str, cfg: Config):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    from inkdet.datasets import augmentations as AA

    assert split in ["train", "val", "test"]
    augmentations: list = []

    if split == "train":

        def _to_augmentations(_augmentations: List[Dict]):
            _aug = []
            for a in _augmentations:
                aug_type: int = a.pop("type")
                params = a.copy()
                if "transforms" in params:
                    params["transforms"] = _to_augmentations(params["transforms"])

                if hasattr(AA, aug_type):
                    _aug.append(getattr(AA, aug_type)(**params))
                else:
                    _aug.append(getattr(A, aug_type)(**params))

            return _aug

        assert len(cfg.train_augmentations) > 0, "No Data Augmentations"
        augmentations += _to_augmentations(cfg.train_augmentations)

    augmentations += [
        A.Normalize(
            mean=[cfg.norm_mean] * cfg.in_channels,
            std=[cfg.norm_std] * cfg.in_channels,
        ),
        ToTensorV2(transpose_mask=True),
    ]

    return A.Compose(augmentations)


def get_train_valid_ids(valid_id: str):
    if valid_id in ["1", "2", "3"]:
        train_fragment_ids = ["1", "2", "3"]
    else:
        train_fragment_ids = FRAGMENTS

    train_ids = [_id for _id in train_fragment_ids if _id != valid_id]
    valid_ids = [valid_id]

    return train_ids, valid_ids


def build_dataloader(
    split: str,
    fragment_ids: List[str],
    cfg: Config,
    pseudo_label_fold_id: Optional[str] = None,
):
    assert split in ["train", "val", "test"]

    patch_size = cfg.patch_size
    stride = cfg.stride
    if split != "train" and cfg.val_patch_size > 0:
        patch_size = cfg.val_patch_size
    if split != "train" and cfg.val_stride > 0:
        stride = cfg.val_stride

    data_dir = "train" if split != "test" else "test"
    images, masks, labels, pseudo_labels, xyxys, ids = load_patched_images(
        data_dir,
        fragment_ids,
        patch_size,
        stride,
        cfg,
        pseudo_label_fold_id,
    )
    xyxys = np.stack(xyxys)

    dataset = SurfaceVolumeDataset(
        cfg=cfg,
        training=split == "train",
        images=images,
        masks=masks if cfg.use_mask else None,
        labels=labels,
        pseudo_labels=pseudo_labels,
        xyxys=xyxys,
        transform=get_transforms(split=split, cfg=cfg),
        task=cfg.task,
    )
    logger.info(f"{split} dataset:\n{dataset}")

    sampler = None
    if split == "train" and cfg.use_weighted_random_sampler:
        logger.info("Enable WeightedRandomSampler")
        ids = np.array(ids)
        weights = np.zeros(len(ids), dtype=np.float32)
        for _id in np.unique(ids):
            logger.info(f"fragment id: w={1 / np.sum(ids == _id)}")
            weights[ids == _id] = 1 / np.sum(ids == _id)

        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=False,
        )
        shuffle = None
    elif split == "train":
        shuffle = True
    else:
        shuffle = False

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=shuffle,
        sampler=sampler,
        worker_init_fn=partial(worker_init_fn, seed=cfg.seed),
        generator=generator,
        pin_memory=True,
        drop_last=split == "train",
    )

    return dataloader, xyxys
