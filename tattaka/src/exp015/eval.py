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
from train import (
    EXP_ID,
    InkDetDataModule,
    InkDetLightningModel,
    fbeta_score,
    find_threshold_percentile,
)

"""
resnetrs50_split3d7x7csn_mixup/fold1: score: 0.5854133858369819, threshold: 0.895898437500001
resnetrs50_split3d7x7csn_mixup/fold2: score: 0.43659195807571055, threshold: 0.9190429687500009
resnetrs50_split3d7x7csn_mixup/fold3: score: 0.5972062582119327, threshold: 0.9259765625000009
resnetrs50_split3d7x7csn_l2_mixup/fold1 :score: 0.5309603341813907, threshold: 0.897949218750001
resnet34d_split3d7x7csn_mixup/fold1: score: 0.5386574886439283, threshold: 0.9041015625000008
resnetrs50_split3d7x7csn_mixup_384/fold1: score: 0.5634811861752591, threshold: 0.8927734375000009
resnetrs50_split3d7x7csn_mixup_384/fold2: score: 0.36933805610328785, threshold: 0.934179687500001
resnetrs50_split3d7x7csn_mixup_384/fold3: score: 0.5923218972107448, threshold: 0.9237304687500009
resnetrs50_split3d7x7csn_mixup_192/fold1: score: 0.5028970241417662, threshold: 0.902734375000001
resnetrs50_split3d7x7csn_mixup_192/fold2: score: 0.4238591298448317, threshold: 0.913378906250001
resnetrs50_split3d7x7csn_mixup_192/fold3: score: 0.5959785143015135, threshold: 0.9300781250000011
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
        ).test_dataloader()

        logdir = f"../../logs/exp{EXP_ID}/{args.logdir}/fold{i}"
        ckpt_path = glob(
            f"{logdir}/**/best_loss.ckpt",
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
            volume, _, x, y = batch
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
        fragment_mask = np.array(
            Image.open(
                f"../../input/vesuvius-challenge-ink-detection/train/{valid_idx}/mask.png"
            ).convert("1")
        )
        count_pix *= fragment_mask
        p_valid /= count_pix
        p_valid = np.nan_to_num(p_valid)
        count_pix = count_pix > 0
        p_valid *= fragment_mask
        p_valid_tmp = p_valid.reshape(-1)[np.where(count_pix.reshape(-1))]
        y_valid_tmp = y_valid.reshape(-1)[np.where(count_pix.reshape(-1))]
        threshold = find_threshold_percentile(y_valid_tmp, p_valid_tmp)
        np.save(f"{logdir}/logits", p_valid)
        p_valid = p_valid > np.quantile(p_valid_tmp, threshold)
        score = fbeta_score(y_valid, p_valid, beta=0.5)
        print(f"{args.logdir}/fold{valid_idx}: score: {score}, threshold: {threshold}")
        np.save(
            f"{logdir}/preds",
            (p_valid > np.quantile(p_valid_tmp, threshold)).astype(int),
        )
        np.save(f"{logdir}/true", y_valid)


if __name__ == "__main__":
    main(get_args())
