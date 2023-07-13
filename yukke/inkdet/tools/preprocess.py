from pathlib import Path
from typing import List

import cv2
from loguru import logger


@logger.catch()
def main():
    root_path = Path("/kaggle/input/vesuvius-challenge-ink-detection/train")
    in_split: str = "2"
    in_path = root_path / in_split

    new_split_a = f"{in_split}a"
    out_path_a = root_path / new_split_a
    out_path_a.mkdir(parents=True, exist_ok=True)
    new_split_b = f"{in_split}b"
    out_path_b = root_path / new_split_b
    out_path_b.mkdir(parents=True, exist_ok=True)

    for filename in ["inklabels.png", "ir.png", "mask.png"]:
        logger.info(f"Processing {filename} ...")
        image = cv2.imread(str(in_path / filename), cv2.IMREAD_UNCHANGED)
        if image.ndim == 2:
            _, w = image.shape
        elif image.ndim == 3:
            _, w, _ = image.shape
            image = image[..., 0]
        else:
            raise ValueError(f"Unexpected image shape: {image}")

        cv2.imwrite(str(out_path_a / filename), image[:, : w // 2])
        cv2.imwrite(str(out_path_b / filename), image[:, w // 2 :])

    for in_tiff_file in sorted((in_path / "surface_volume").glob("*.tif")):
        logger.info(f"Processing {in_tiff_file.name} ...")
        image = cv2.imread(str(in_tiff_file), cv2.IMREAD_UNCHANGED)
        _, w = image.shape

        out_tiff_file_a = root_path / new_split_a / "surface_volume" / in_tiff_file.name
        out_tiff_file_a.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_tiff_file_a), image[:, : w // 2])

        out_tiff_file_b = root_path / new_split_b / "surface_volume" / in_tiff_file.name
        out_tiff_file_b.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_tiff_file_b), image[:, w // 2 :])


# @logger.catch()
# def main():
#     root_path = Path("/kaggle/input/vesuvius-challenge-ink-detection/train")
#     in_split: str = "2"
#     in_path = root_path / in_split

#     new_split_1 = f"{in_split}_1"
#     out_path_1 = root_path / new_split_1
#     out_path_1.mkdir(parents=True, exist_ok=True)
#     new_split_2 = f"{in_split}_2"
#     out_path_2 = root_path / new_split_2
#     out_path_2.mkdir(parents=True, exist_ok=True)
#     new_split_3 = f"{in_split}_3"
#     out_path_3 = root_path / new_split_3
#     out_path_3.mkdir(parents=True, exist_ok=True)

#     for filename in ["inklabels.png", "ir.png", "mask.png"]:
#         logger.info(f"Processing {filename} ...")
#         image = cv2.imread(str(in_path / filename), cv2.IMREAD_UNCHANGED)
#         if image.ndim == 2:
#             h, _ = image.shape
#         elif image.ndim == 3:
#             h, _, _ = image.shape
#             image = image[..., 0]
#         else:
#             raise ValueError(f"Unexpected image shape: {image}")

#         cv2.imwrite(str(out_path_1 / filename), image[: h // 3])
#         cv2.imwrite(str(out_path_2 / filename), image[h // 3 : 2 * h // 3])
#         cv2.imwrite(str(out_path_3 / filename), image[2 * h // 3 :])

#     for in_tiff_file in sorted((in_path / "surface_volume").glob("*.tif")):
#         logger.info(f"Processing {in_tiff_file.name} ...")
#         image = cv2.imread(str(in_tiff_file), cv2.IMREAD_UNCHANGED)
#         h, _ = image.shape

#         out_tiff_file_1 = out_path_1 / "surface_volume" / in_tiff_file.name
#         out_tiff_file_1.parent.mkdir(parents=True, exist_ok=True)
#         cv2.imwrite(str(out_tiff_file_1), image[: h // 3])

#         out_tiff_file_2 = out_path_2 / "surface_volume" / in_tiff_file.name
#         out_tiff_file_2.parent.mkdir(parents=True, exist_ok=True)
#         cv2.imwrite(str(out_tiff_file_2), image[h // 3 : 2 * h // 3])

#         out_tiff_file_3 = out_path_3 / "surface_volume" / in_tiff_file.name
#         out_tiff_file_3.parent.mkdir(parents=True, exist_ok=True)
#         cv2.imwrite(str(out_tiff_file_3), image[2 * h // 3 :])


if __name__ == "__main__":
    main()
