from typing import Iterable, Optional, Tuple, Union

import albumentations as A
import numpy as np


def random_erasing(
    img: np.ndarray,
    min_fill_value: int,
    max_fill_value: int,
    holes: Iterable[Tuple[int, int, int, int]],
) -> np.ndarray:
    img = img.copy()  # make a copy not to modify the original image
    c: int = img.shape[-1]
    for x1, y1, x2, y2 in holes:
        img[y1:y2, x1:x2] = np.random.randint(min_fill_value, max_fill_value, (y2 - y1, x2 - x1, c))
    return img


class RandomErasing(A.CoarseDropout):
    def __init__(
        self,
        max_holes: int = 8,
        max_height: int = 8,
        max_width: int = 8,
        min_holes: Optional[int] = None,
        min_height: Optional[int] = None,
        min_width: Optional[int] = None,
        fill_value: int = 255,
        mask_fill_value: Optional[int] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(RandomErasing, self).__init__(
            max_holes=max_holes,
            max_height=max_height,
            max_width=max_width,
            min_holes=min_holes,
            min_height=min_height,
            min_width=min_width,
            fill_value=fill_value,
            mask_fill_value=mask_fill_value,
            always_apply=always_apply,
            p=p,
        )

    def apply(
        self,
        img: np.ndarray,
        fill_value: Union[int, float],
        holes: Iterable[Tuple[int, int, int, int]],
        **params,
    ) -> np.ndarray:
        return random_erasing(img, 0, fill_value, holes)

    def get_transform_init_args_names(self):
        return (
            "max_holes",
            "max_height",
            "max_width",
            "min_holes",
            "min_height",
            "min_width",
            "fill_value",
            "mask_fill_value",
        )
