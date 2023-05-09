import argparse
import datetime
import math
import warnings
from functools import partial
from glob import glob
from typing import List

import numpy as np
import PIL.Image as Image
import pytorch_lightning as pl
import torch
from scipy.optimize import minimize
from torch.nn import functional as F
from tqdm import tqdm
from train import EXP_ID, InkDetDataModule, InkDetLightningModel, fbeta_score

"""

"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def func_percentile(y_true: np.ndarray, y_pred: np.ndarray, t: List[float]):
    score = fbeta_score(
        y_true.flatten(),
        (y_pred > np.quantile(y_pred, np.clip(t[0], 0, 1), axis=(1, 3), keepdims=True))
        .astype(int)
        .flatten(),
        beta=0.5,
    )
    return -score


def find_threshold_percentile(y_true: np.ndarray, y_pred: np.ndarray):
    x0 = [0.5]
    threshold = minimize(
        partial(
            func_percentile,
            y_true,
            y_pred,
        ),
        x0,
        method="nelder-mead",
    ).x[0]
    return np.clip(threshold, 0, 1)


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
            break
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
        div_h, div_w = 100, 100
        new_h = (math.ceil(p_valid.shape[0] / div_h)) * div_h
        new_w = (math.ceil(p_valid.shape[1] / div_w)) * div_w
        new_p_valid = np.zeros((new_h, new_w))
        new_count_pix = np.zeros((new_h, new_w))
        new_y_valid = np.zeros((new_h, new_w))
        new_p_valid[: p_valid.shape[0], : p_valid.shape[1]] = p_valid
        new_count_pix[: p_valid.shape[0], : p_valid.shape[1]] = count_pix
        new_y_valid[: p_valid.shape[0], : p_valid.shape[1]] = y_valid

        new_count_pix = new_count_pix.reshape(
            div_h, new_h // div_h, div_w, new_w // div_w
        )
        p_valid_tmp = (
            new_p_valid.reshape(div_h, new_h // div_h, div_w, new_w // div_w)
        )[
            new_count_pix.sum((1, 3), keepdims=True).expand(
                -1, new_h // div_h, -1, new_w // div_w
            )
            > 0
        ]
        y_valid_tmp = (
            new_y_valid.reshape(div_h, new_h // div_h, div_w, new_w // div_w)
        )[
            new_count_pix.sum((1, 3), keepdims=True).expand(
                -1, new_h // div_h, -1, new_w // div_w
            )
            > 0
        ]

        threshold = find_threshold_percentile(y_valid_tmp, p_valid_tmp)
        np.save(f"{logdir}/logits_v2", p_valid)
        p_valid = (
            p_valid_tmp
            > np.quantile(p_valid_tmp, threshold, axis=(1, 3), keepdims=True)
        ).reshape(new_h, new_w)[: p_valid.shape[0], : p_valid.shape[1]]
        score = fbeta_score(y_valid, p_valid, beta=0.5)
        print(f"{args.logdir}/fold{valid_idx}: score: {score}, threshold: {threshold}")
        np.save(
            f"{logdir}/preds_v2",
            (p_valid > np.quantile(p_valid_tmp, threshold)).astype(int),
        )
        np.save(f"{logdir}/true", y_valid)


if __name__ == "__main__":
    main(get_args())
