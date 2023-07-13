import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from inkdet.utils.config import Config


class SurfaceVolumeDataset(Dataset):
    def __init__(
        self,
        cfg: Config,
        training: bool,
        images: np.ndarray,
        masks: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        pseudo_labels: Optional[np.ndarray] = None,
        xyxys: Optional[Tuple[int, int, int, int]] = None,
        transform: Optional[List[Dict]] = None,
        task: str = "segmentation",
    ):
        assert task in ["segmentation", "classification", "segmentation-classification"]
        self.training = training
        self.cfg = cfg
        self.images = images
        self.labels = labels
        self.pseudo_labels = pseudo_labels
        self.transform = transform
        self.xyxys = xyxys
        self.task = task

        if labels is not None:
            assert len(images) == len(labels), f"{len(images)} != {len(labels)}"

        if pseudo_labels is not None:
            assert len(images) == len(pseudo_labels), f"{len(images)} != {len(pseudo_labels)}"

        if masks is not None:
            assert len(images) == len(masks), f"{len(images)} != {len(masks)}"

            mask_idx = [i for i, mask in enumerate(masks) if mask.sum() > 0]
            self.images = [self.images[i] for i in mask_idx]
            if labels is not None:
                self.labels = [self.labels[i] for i in mask_idx]
            if pseudo_labels is not None:
                self.pseudo_labels = [self.pseudo_labels[i] for i in mask_idx]
            if xyxys is not None:
                self.xyxys = [self.xyxys[i] for i in mask_idx]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.labels is not None:
            image: np.ndarray = self.images[idx]  # (H, W, C)
            label: np.ndarray = self.labels[idx]  # (H, W, 1)
            if self.pseudo_labels is not None:
                pseudo_label: np.ndarray = self.pseudo_labels[idx]  # (H, W, 1)
            else:
                pseudo_label = None

            if self.transform:
                if pseudo_label is not None:
                    data = self.transform(image=image, mask=np.concatenate([label, pseudo_label], axis=-1))
                else:
                    data = self.transform(image=image, mask=label)
                image = data["image"]  # (C, H, W)
                label = data["mask"]  # (N, H, W): N = 1 or 2

            if self.task == "segmentation":
                return image, label
            elif self.task == "classification":
                rot_k = np.random.choice(range(4)) if self.training else idx % 4
                image = torch.rot90(image, rot_k, dims=(-2, -1))
                label = 1 if rot_k in [1, 3] else 0
                return image, torch.tensor([label], dtype=torch.float32)
            elif self.task == "segmentation-classification":
                rot_k = random.randint(0, 3) if self.training else 0
                # rot_k = random.randint(0, 3) if self.training else idx % 4
                image = torch.rot90(image, rot_k, dims=(-2, -1))
                label = torch.rot90(label, rot_k, dims=(-2, -1))
                # 0: none or rot180, 1: rot90 or rot270
                rot_label = torch.tensor([1 if rot_k in [1, 3] else 0], dtype=torch.float32)
                return image, label, rot_label
        else:
            image = self.images[idx]

            if self.transform:
                data = self.transform(image=image)
                image = data["image"]

            return image

    def __repr__(self):
        repr = __class__.__name__ + "\n"
        repr += f"len(image) = {len(self.images)}\n"
        repr += f"transform = {self.transform}"
        return repr
