import argparse
import datetime
import warnings
from functools import partial
from glob import glob
from typing import List

import cv2
import numpy as np
import PIL.Image as Image
import pytorch_lightning as pl
import torch
from scipy.optimize import minimize
from torch.nn import functional as F
from tqdm import tqdm
from train import EXP_ID, InkDetDataModule, InkDetLightningModel, fbeta_score

"""
resnetrs50_split3d5x7csn_mixup/fold1: score: 0.6179195625960188, threshold: 0.9078125000000009, min_size: 0.7124999999999997
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def postprocess(y_pred: np.ndarray, use_area: np.ndarray, t: float):
    y_pred = (y_pred > np.quantile(use_area, np.clip(t, 0, 1))).astype(int)
    return y_pred


def delete_min_area(y_pred: np.ndarray, t: float):
    num_component, component = cv2.connectedComponents(y_pred.astype(np.uint8))
    size_list = np.array([(c == component) for c in range(1, num_component)])
    size_sum = size_list.sum((1, 2))
    min_size = np.quantile(size_sum, np.clip(t, 0, 1))
    y_pred = size_list[size_sum > min_size].sum(0)
    return y_pred


def find_threshold_percentile(
    y_true: np.ndarray, y_pred: np.ndarray, use_area: np.ndarray
):
    def func_percentile(
        y_true: np.ndarray, y_pred: np.ndarray, use_area: np.ndarray, t: List[float]
    ):
        score = fbeta_score(
            y_true,
            postprocess(y_pred, use_area, t[0]),
            beta=0.5,
        )
        return -score

    x0 = [0.5]
    threshold = minimize(
        partial(
            func_percentile,
            y_true,
            y_pred,
            use_area,
        ),
        x0,
        method="nelder-mead",
    ).x[0]
    return np.clip(threshold, 0, 1)


def find_min_size_percentile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    use_area: np.ndarray,
    threshold: np.ndarray,
):
    x0 = 1.0

    def func_percentile(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        use_area: np.ndarray,
        t1: float,
        t2: float,
    ):
        y_pred = postprocess(y_pred, use_area, t1)

        score = fbeta_score(
            y_true,
            delete_min_area(y_pred, t2),
            beta=0.5,
        )
        return -score

    # best_score = 0
    # while True:
    #     score = func_percentile(y_true, y_pred, use_area, threshold, x0)
    #     if best_score > score:
    #         best_score = score
    #         x0 -= 0.005
    #     else:
    #         break
    # min_size = x0
    min_size = minimize(
        partial(
            func_percentile,
            y_true,
            y_pred,
            use_area,
            threshold,
        ),
        x0,
        method="nelder-mead",
    ).x[0]
    return np.clip(min_size, 0, 1)


def get_args() -> argparse.Namespace:
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--seed",
        default=2022,
        type=int,
        metavar="SE",
        help="seed number",
        dest="seed",
    )
    dt_now = datetime.datetime.now()
    parent_parser.add_argument(
        "--logdir",
        default=f"{dt_now.strftime('%Y%m%d-%H-%M-%S')}",
    )
    parent_parser.add_argument(
        "--fold",
        type=int,
        default=0,
    )
    parser = InkDetDataModule.add_model_specific_args(parent_parser)
    return parser.parse_args()


def main(args):
    pl.seed_everything(args.seed)
    warnings.simplefilter("ignore")
    fragment_ids = [1, 2, 3, 4, 5]
    for i, valid_idx in enumerate(fragment_ids):
        if args.fold != i:
            continue
        valid_volume_paths = np.concatenate(
            [
                np.asarray(
                    sorted(
                        glob(
                            f"../../input/vesuvius_patches_32_5fold/train/{fragment_id}/surface_volume/**/*.npy",
                            recursive=True,
                        )
                    )
                )
                for fragment_id in fragment_ids
                if fragment_id == valid_idx
            ]
        )

        dataloader = InkDetDataModule(
            train_volume_paths=valid_volume_paths,
            valid_volume_paths=valid_volume_paths,
            image_size=args.image_size,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            preprocess_in_model=True,
        ).test_dataloader()

        logdir = f"../../logs/exp{EXP_ID}/{args.logdir}/fold{i}"
        ckpt_path = glob(
            f"{logdir}/**/best_fbeta.ckpt",
            recursive=True,
        )[0]
        print(f"ckpt_path = {ckpt_path}")
        model = InkDetLightningModel.load_from_checkpoint(
            ckpt_path,
            valid_fragment_id=valid_idx,
            pretrained=False,
            preprocess_in_model=True,
        )
        model.eval()
        model = model.to(device=device)
        y_valid = np.array(
            Image.open(
                f"../../input/vesuvius-challenge-ink-detection-5fold/train/{valid_idx}/inklabels.png"
            ).convert("1")
        )
        p_valid = np.zeros_like(y_valid, dtype=np.float32)
        count_pix = np.zeros_like(y_valid, dtype=np.float32)
        tta_set = [
            "vanilla",
            "flip_v",
            "flip_h",
            "flip_vh",
            "flip_c",
            "flip_vc",
            "flip_hc",
            "flip_vhc",
        ]
        for batch in tqdm(dataloader):
            volume, _, x, y = batch
            for i in range(len(volume)):
                if i % len(tta_set) == 1:
                    volume[i] = volume[i].flip(1)
                elif i % len(tta_set) == 2:
                    volume[i] = volume[i].flip(2)
                elif i % len(tta_set) == 3:
                    volume[i] = volume[i].flip(1).flip(2)
                elif i % len(tta_set) == 4:
                    volume[i] = volume[i].flip(0)
                elif i % len(tta_set) == 5:
                    volume[i] = volume[i].flip(0).flip(1)
                elif i % len(tta_set) == 6:
                    volume[i] = volume[i].flip(0).flip(2)
                elif i % len(tta_set) == 7:
                    volume[i] = volume[i].flip(0).flip(1).flip(2)

            volume = volume.to(device)
            with torch.no_grad():
                pred_batch = torch.sigmoid(model.model(volume))

            for i in range(len(pred_batch)):
                if i % len(tta_set) == 1:
                    pred_batch[i] = pred_batch[i].flip(1)
                elif i % len(tta_set) == 2:
                    pred_batch[i] = pred_batch[i].flip(2)
                elif i % len(tta_set) == 3:
                    pred_batch[i] = pred_batch[i].flip(1).flip(2)
                elif i % len(tta_set) == 4:
                    pred_batch[i] = pred_batch[i].flip(0)
                elif i % len(tta_set) == 5:
                    pred_batch[i] = pred_batch[i].flip(0).flip(1)
                elif i % len(tta_set) == 6:
                    pred_batch[i] = pred_batch[i].flip(0).flip(2)
                elif i % len(tta_set) == 7:
                    pred_batch[i] = pred_batch[i].flip(0).flip(1).flip(2)

            pred_batch = F.interpolate(
                pred_batch.detach().to(torch.float32).cpu(),
                scale_factor=32,
                mode="bilinear",
                align_corners=True,
            ).numpy()
            for xi, yi, pi in zip(
                x,
                y,
                pred_batch,
            ):
                y_lim, x_lim = y_valid[
                    yi * 32 : yi * 32 + volume.shape[-2],
                    xi * 32 : xi * 32 + volume.shape[-1],
                ].shape
                p_valid[
                    yi * 32 : yi * 32 + volume.shape[-2],
                    xi * 32 : xi * 32 + volume.shape[-1],
                ] += pi[0, :y_lim, :x_lim]
                count_pix[
                    yi * 32 : yi * 32 + volume.shape[-2],
                    xi * 32 : xi * 32 + volume.shape[-1],
                ] += np.ones_like(pi[0, :y_lim, :x_lim])
        fragment_mask = np.array(
            Image.open(
                f"../../input/vesuvius-challenge-ink-detection-5fold/train/{valid_idx}/mask.png"
            ).convert("1")
        )
        count_pix *= fragment_mask
        p_valid /= count_pix
        p_valid = np.nan_to_num(p_valid)
        count_pix = count_pix > 0
        p_valid *= fragment_mask
        p_valid_tmp = p_valid.reshape(-1)[np.where(count_pix.reshape(-1))]
        print("Start optimizing top threshold.....")
        threshold = find_threshold_percentile(y_valid, p_valid, p_valid_tmp)
        print("Finish optimizing top threshold!")
        print("Start optimizing min_size.....")
        min_size = find_min_size_percentile(y_valid, p_valid, p_valid_tmp, threshold)
        print("Finish optimizing min_size!")
        p_valid = delete_min_area(
            y_pred=postprocess(y_pred=p_valid, use_area=p_valid_tmp, t=threshold),
            t=min_size,
        )
        score = fbeta_score(y_valid, p_valid, beta=0.5)
        print(
            f"{args.logdir}/fold{valid_idx}: score: {score}, threshold: {threshold}, min_size: {min_size}"
        )
        np.save(
            f"{logdir}/preds_pp",
            p_valid.astype(int),
        )


if __name__ == "__main__":
    main(get_args())
