import argparse
import datetime
import warnings
from glob import glob

import numpy as np
import PIL.Image as Image
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from tqdm import tqdm
from train import EXP_ID, InkDetDataModule, InkDetLightningModel, dice, find_threshold

"""
fold 0: score: 0.49920214799331863, threshold: 0.25
# 128: score: 0.4641925060191766, threshold: 0.27
fold 2: score: 0.5349645846748229, threshold: 0.38
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


def main(args):
    pl.seed_everything(args.seed)
    warnings.simplefilter("ignore")
    fragment_ids = [1, 2, 3]
    for i, valid_idx in enumerate(fragment_ids):
        if args.fold != i:
            continue
        valid_volume_paths = np.concatenate(
            [
                np.asarray(
                    sorted(
                        glob(
                            f"../../input/vesuvius_patches_32/train/{fragment_id}/surface_volume/**/*.npy",
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
        ).val_dataloader()

        logdir = f"../../logs/exp{EXP_ID}/{args.logdir}/fold{i}"
        ckpt_path = glob(
            f"{logdir}/**/best_dice.ckpt",
            recursive=True,
        )[0]
        print(f"ckpt_path = {ckpt_path}")
        model = InkDetLightningModel.load_from_checkpoint(
            ckpt_path, valid_fragment_id=valid_idx, pretrained=False
        )
        model.eval()
        model = model.to(device=device)
        y_valid = model.y_valid
        p_valid = model.p_valid
        count_pix = model.count_pix
        for batch in tqdm(dataloader):
            volume, _, x, y, _ = batch
            volume = volume.to(device)
            with torch.no_grad():
                pred_batch = model.model(volume)
            pred_batch = F.interpolate(
                torch.sigmoid(pred_batch).detach().to(torch.float32).cpu(),
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
        p_valid /= count_pix
        p_valid = np.nan_to_num(p_valid)
        count_pix = count_pix > 0
        fragment_mask = np.array(
            Image.open(
                f"../../input/vesuvius-challenge-ink-detection/train/{valid_idx}/mask.png"
            ).convert("1")
        )
        p_valid *= fragment_mask
        threshold = find_threshold(p_valid, y_valid, count_pix)
        p_valid = p_valid > threshold
        score = dice(p_valid, y_valid, count_pix)
        print(f"score: {score}, threshold: {threshold}")
        np.save(f"{logdir}/debug_p_valid", p_valid)
        np.save(f"{logdir}/debug_y_valid", y_valid)


if __name__ == "__main__":
    main(get_args())
