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
    InkDetLightningModel3d,
    fbeta_score,
    find_threshold_percentile,
)

"""
####### w/o postprocess ##########
resnet3d50csnir1x32_mixup_ep30/fold1: score: 0.652337684867544, threshold: 0.8851562500000008
resnet3d50csnir1x32_mixup_ep30/fold2: score: 0.7004049381518501, threshold: 0.9148437500000011
resnet3d50csnir1x32_mixup_ep30/fold3: score: 0.6985949376703707, threshold: 0.920703125000001
resnet3d50csnir1x32_mixup_ep30/fold4: score: 0.7807912449768651, threshold: 0.8811523437500008
resnet3d50csnir1x32_mixup_ep30/fold5: score: 0.7364386200206916, threshold: 0.8951171875000009

resnet3d152csnir1x24_mixup_ep30/fold1: score: 0.6601534242104694, threshold: 0.9004882812500009
resnet3d152csnir1x24_mixup_ep30/fold2: score: 0.7275147505570198, threshold: 0.8898437500000009
resnet3d152csnir1x24_mixup_ep30/fold3: score: 0.7139278750354907, threshold: 0.925097656250001
resnet3d152csnir1x24_mixup_ep30/fold4: score: 0.7900112110659766, threshold: 0.8844726562500008
resnet3d152csnir1x24_mixup_ep30/fold5: score: 0.7374516137096395, threshold: 0.893359375000001
####### w/o postprocess ##########
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
            image_size=256,
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
        model = InkDetLightningModel3d.load_from_checkpoint(
            ckpt_path,
            valid_fragment_id=valid_idx,
            pretrained=False,
            preprocess_in_model=True,
        )
        model.eval()
        model = model.half().to(device=device)
        y_valid = np.array(
            Image.open(
                f"../../input/vesuvius-challenge-ink-detection-5fold/train/{valid_idx}/inklabels.png"
            ).convert("1")
        )
        p_valid = np.zeros_like(y_valid, dtype=np.float16)
        count_pix = np.zeros_like(y_valid, dtype=np.uint8)
        tta_set = [
            "vanilla",
            "flip_v",
            "flip_h",
            "flip_vh",
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
            pad = (256 - args.image_size) // 2
            if pad > 0:
                volume_new = volume[:, :, pad:-pad, pad:-pad].to(device)
            else:
                volume_new = volume.to(device)
            with torch.no_grad():
                pred_batch = torch.sigmoid(model.model_ema.module(volume_new.half()))

            for i in range(len(pred_batch)):
                if i % len(tta_set) == 1:
                    pred_batch[i] = pred_batch[i].flip(1)
                elif i % len(tta_set) == 2:
                    pred_batch[i] = pred_batch[i].flip(2)
                elif i % len(tta_set) == 3:
                    pred_batch[i] = pred_batch[i].flip(1).flip(2)

            pred_batch = (
                F.interpolate(
                    pred_batch.detach().to(torch.float32).cpu(),
                    scale_factor=32,
                    mode="bilinear",
                    align_corners=True,
                )
                .to(torch.float16)
                .numpy()
            )
            pred_batch_new = np.zeros(
                list(pred_batch.shape[:2]) + list(volume.shape[-2:])
            )  # [bs, 1] + [w, h]
            if pad > 0:
                pred_batch_new[:, :, pad:-pad, pad:-pad] = pred_batch
            else:
                pred_batch_new = pred_batch
            for xi, yi, pi in zip(
                x,
                y,
                pred_batch_new,
            ):
                y_lim, x_lim = y_valid[
                    yi * 32 : yi * 32 + volume.shape[-2],
                    xi * 32 : xi * 32 + volume.shape[-1],
                ].shape
                p_valid[
                    yi * 32 : yi * 32 + volume.shape[-2],
                    xi * 32 : xi * 32 + volume.shape[-1],
                ] += pi[0, :y_lim, :x_lim]
                count_pix_single = np.zeros_like(pi[0], dtype=np.uint8)
                if pad > 0:
                    count_pix_single[pad:-pad, pad:-pad] = np.ones_like(
                        pred_batch[0][0], dtype=np.uint8
                    )
                else:
                    count_pix_single = np.ones_like(pred_batch[0][0], dtype=np.uint8)
                count_pix[
                    yi * 32 : yi * 32 + volume.shape[-2],
                    xi * 32 : xi * 32 + volume.shape[-1],
                ] += count_pix_single[:y_lim, :x_lim]
        fragment_mask = (
            np.array(
                Image.open(
                    f"../../input/vesuvius-challenge-ink-detection-5fold/train/{valid_idx}/mask.png"
                ).convert("1")
            )
            > 0
        )
        count_pix *= fragment_mask
        p_valid /= count_pix
        p_valid = np.nan_to_num(p_valid, posinf=0, neginf=0)
        count_pix = count_pix > 0
        p_valid *= fragment_mask
        p_valid_tmp = p_valid.reshape(-1)[np.where(count_pix.reshape(-1))]
        y_valid_tmp = y_valid.reshape(-1)[np.where(count_pix.reshape(-1))]
        threshold = find_threshold_percentile(y_valid_tmp, p_valid_tmp)
        np.save(f"{logdir}/logits_tta_fp16", p_valid)
        p_valid = p_valid > np.quantile(p_valid_tmp, threshold)
        score = fbeta_score(y_valid, p_valid, beta=0.5)
        print(f"{args.logdir}/fold{valid_idx}: score: {score}, threshold: {threshold}")
        np.save(
            f"{logdir}/preds_tta_fp16",
            (p_valid > np.quantile(p_valid_tmp, threshold)).astype(int),
        )
        np.save(f"{logdir}/true", y_valid)


if __name__ == "__main__":
    main(get_args())
