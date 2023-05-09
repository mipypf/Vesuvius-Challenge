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
resnetrs50_split3d7x7csn_mixup/fold1: score: 0.5840947316990723, threshold: 0.901171875000001
resnetrs50_split3d7x7csn_mixup/fold2: score: 0.644994005660861, threshold: 0.907812500000001
resnetrs50_split3d7x7csn_mixup/fold3: score: 0.5780603014364925, threshold: 0.928027343750001
resnetrs50_split3d7x7csn_mixup/fold4: score: 0.6837184367493706, threshold: 0.8910156250000009
resnetrs50_split3d7x7csn_mixup/fold5: score: 0.7033885760711703, threshold: 0.8988281250000009

resnetrs50_split3d6x7csn_mixup/fold1: score: 0.567010828905749, threshold: 0.898046875000001
resnetrs50_split3d6x7csn_mixup/fold2: score: 0.6587065643894002, threshold: 0.910546875000001
resnetrs50_split3d6x7csn_mixup/fold3: score: 0.5944430299923158, threshold: 0.9311523437500009
resnetrs50_split3d6x7csn_mixup/fold4: score: 0.6986253200372938, threshold: 0.8851562500000008
resnetrs50_split3d6x7csn_mixup/fold5: score: 0.7030854611641069, threshold: 0.8983398437500009

resnetrs50_split3d6x6csn_mixup/fold1: score: 0.5546279234875113, threshold: 0.909570312500001
resnetrs50_split3d6x6csn_mixup/fold2: score: 0.6631772574744388, threshold: 0.9041015625000008
resnetrs50_split3d6x6csn_mixup/fold3: score: 0.5700231470041768, threshold: 0.9335937500000009
resnetrs50_split3d6x6csn_mixup/fold4: score: 0.69033207071837, threshold: 0.8868164062500009
resnetrs50_split3d6x6csn_mixup/fold5: score: 0.7106331345516264, threshold: 0.8966796875000009

resnetrs50_split3d5x7csn_mixup/fold1: score: 0.5805118259687848, threshold: 0.896093750000001
resnetrs50_split3d5x7csn_mixup/fold2: sco   re: 0.676733279214622, threshold: 0.9027343750000009
resnetrs50_split3d5x7csn_mixup/fold3: score: 0.5906556470402545, threshold: 0.932226562500001
resnetrs50_split3d5x7csn_mixup/fold4: score: 0.7072238621930932, threshold: 0.8811523437500008
resnetrs50_split3d5x7csn_mixup/fold5: score: 0.7142429057127576, threshold: 0.9019531250000009

# resnetrs50_split3d3x16csn_mixup/fold1: score: 0.5675262962986994, threshold: 0.903906250000001
# resnetrs50_split3d5x7csn_s4_mixup/fold3: score: 0.6001588493663199, threshold: 0.9313476562500009

resnetrs50_split3d5x9csn_mixup/fold1: score: 0.565857857646758, threshold: 0.9045898437500008
resnetrs50_split3d5x9csn_mixup/fold2: score: 0.6794888739521351, threshold: 0.9015625000000009
resnetrs50_split3d5x9csn_mixup/fold3: score: 0.5807587801971041, threshold: 0.9220703125000009
resnetrs50_split3d5x9csn_mixup/fold4: score: 0.713984329894918, threshold: 0.877343750000001
resnetrs50_split3d5x9csn_mixup/fold5: score: 0.7151308799737465, threshold: 0.9003906250000009

resnetrs50_split3d4x9csn_mixup/fold1: score: 0.38284986708566887, threshold: 0.8369140625000009
resnetrs50_split3d4x9csn_mixup/fold2: score: 0.6571024717686051, threshold: 0.9068359375000008
resnetrs50_split3d4x9csn_mixup/fold3: score: 0.5847637458480953, threshold: 0.9289062500000009
resnetrs50_split3d4x9csn_mixup/fold4: score: 0.713638985286127, threshold: 0.8878906250000008
resnetrs50_split3d4x9csn_mixup/fold5: score: 0.7295468596928188, threshold: 0.8925781250000009


resnetrs50_split3d3x9csn_mixup/fold1: score: 0.5654274680228007, threshold: 0.8994140625000009
resnetrs50_split3d3x9csn_mixup/fold2: score: 0.6554422186662509, threshold: 0.907617187500001
resnetrs50_split3d3x9csn_mixup/fold3: score: 0.6338300055593009, threshold: 0.9259765625000009
resnetrs50_split3d3x9csn_mixup/fold4: score: 0.7234870215828206, threshold: 0.8820312500000009
resnetrs50_split3d3x9csn_mixup/fold5: score: 0.730124363927394, threshold: 0.8994140625000009
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
