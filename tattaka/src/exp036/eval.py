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
resnetrs50_split3d5x7csn_mixup/fold1: score: 0.6131710463848015, threshold: 0.9000000000000009
resnetrs50_split3d5x7csn_mixup/fold2: score: 0.6691522570290931, threshold: 0.908984375000001
resnetrs50_split3d5x7csn_mixup/fold3: score: 0.6352648403441427, threshold: 0.9304687500000011
resnetrs50_split3d5x7csn_mixup/fold4: score: 0.7117744228626159, threshold: 0.8904296875000008
resnetrs50_split3d5x7csn_mixup/fold5: score: 0.7183981581434672, threshold: 0.897265625000001

resnetrs50_split3d3x9csn_l6_mixup/fold1: score: 0.5616668027749042, threshold: 0.9064453125000009
resnetrs50_split3d3x9csn_l6_mixup/fold2: score: 0.6613327992076937, threshold: 0.9031250000000008
resnetrs50_split3d3x9csn_l6_mixup/fold3: score: 0.62ぐらい
resnetrs50_split3d3x9csn_l6_mixup/fold4: score: 0.7222121016589873, threshold: 0.8898437500000009
resnetrs50_split3d3x9csn_l6_mixup/fold5: score: 0.726496356079917, threshold: 0.9018554687500009

ecaresnet50t_split3d5x7csn_mixup/fold1: score: 0.5991458131227679, threshold: 0.9003906250000009
ecaresnet50t_split3d5x7csn_mixup/fold3: score: 0.5937196579164629, threshold: 0.9275390625000011

ecaresnet50t_split3d3x9csn_l6_mixup/fold1: score: 0.5924508528942055, threshold: 0.890039062500001
ecaresnet50t_split3d3x9csn_l6_mixup/fold3: score: 0.6272542699863115, threshold: 0.9288085937500009

resnetrs101_split3d5x7csn_mixup/fold1: score: 0.5990584400180661, threshold: 0.9039062500000008
resnetrs101_split3d5x7csn_mixup/fold3: score: 0.6530235399559506, threshold: 0.9378906250000012

resnetrs101_split3d3x9csn_l6_mixup/fold1: score: 0.568088181745679, threshold: 0.8923828125000008
resnetrs101_split3d3x9csn_l6_mixup/fold3: score: 0.6350392538805103, threshold: 0.930371093750001
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
                f"../../input/vesuvius-challenge-ink-detection-5fold/train/{valid_idx}/mask.png"
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
