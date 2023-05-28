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
####### w/o postprocess ##########
resnetrs50_split3d5x7csn_mixup_ep30/fold1: score: 0.6616110924110317, threshold: 0.896289062500001
resnetrs50_split3d5x7csn_mixup_ep30/fold2: score: 0.6766739264738048, threshold: 0.896582031250001
resnetrs50_split3d5x7csn_mixup_ep30/fold3: score: 0.6893144750045893, threshold: 0.924218750000001
resnetrs50_split3d5x7csn_mixup_ep30/fold4: score: 0.7776853733102816, threshold: 0.8797851562500009
resnetrs50_split3d5x7csn_mixup_ep30/fold5: score: 0.7456068233518977, threshold: 0.894921875000001

resnetrs50_split3d3x9csn_l6_mixup_ep30/fold1: score: 0.6475739401506333, threshold: 0.8937500000000009
resnetrs50_split3d3x9csn_l6_mixup_ep30/fold2: score: 0.7226838982914648, threshold: 0.8966796875000009
resnetrs50_split3d3x9csn_l6_mixup_ep30/fold3: score: 0.7112823609332312, threshold: 0.9267578125000009
resnetrs50_split3d3x9csn_l6_mixup_ep30/fold4: score: 0.7807499937074391, threshold: 0.8776367187500009
resnetrs50_split3d3x9csn_l6_mixup_ep30/fold5: score: 0.7287241432138931, threshold: 0.8927734375000009

convnext_tiny_split3d5x7csn_mixup_ep30/fold1: score: 0.6480399318093204, threshold: 0.8992187500000008
convnext_tiny_split3d5x7csn_mixup_ep30/fold2: score: 0.7349395208454771, threshold: 0.901367187500001
convnext_tiny_split3d5x7csn_mixup_ep30/fold3: score: 0.6900721993798925, threshold: 0.930175781250001
convnext_tiny_split3d5x7csn_mixup_ep30/fold4: score: 0.7887197902193216, threshold: 0.8742187500000009
convnext_tiny_split3d5x7csn_mixup_ep30/fold5: score: 0.7579462384855769, threshold: 0.8944335937500009

convnext_tiny_split3d3x9csn_l6_mixup_ep30/fold1: score: 0.6609206181251918, threshold: 0.8943359375000008
convnext_tiny_split3d3x9csn_l6_mixup_ep30/fold2: score: 0.7103011197961271, threshold: 0.8947265625000009
convnext_tiny_split3d3x9csn_l6_mixup_ep30/fold3: score: 0.7002915696210247, threshold: 0.933105468750001
convnext_tiny_split3d3x9csn_l6_mixup_ep30/fold4: score: 0.7907009230776294, threshold: 0.8817382812500009
convnext_tiny_split3d3x9csn_l6_mixup_ep30/fold5: score: 0.754126946565578, threshold: 0.893652343750001

swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/fold1: score: 0.659608109675679, threshold: 0.895117187500001
swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/fold2: score: 0.7202747225205692, threshold: 0.8996093750000009
swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/fold3: score: 0.7005684116948904, threshold: 0.930078125000001
swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/fold4: score: 0.7830606810622647, threshold: 0.8776367187500009
swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/fold5: score: 0.7512288835151714, threshold: 0.8962890625000011

swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/fold1: score: 0.6493931321137051, threshold: 0.901953125000001
swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/fold2: score: 0.7385469428581766, threshold: 0.9033203125000009
swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/fold3: score: 0.7156795923483393, threshold: 0.9275390625000011
swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/fold4: score: 0.7809338279543435, threshold: 0.8759765625000009
swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/fold5: score: 0.7565529723487409, threshold: 0.892968750000001

swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/fold1: score: 0.6437059820900941, threshold: 0.897265625000001
swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/fold2: score: 0.71618750126515, threshold: 0.9080078125000008
swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/fold3: score: 0.7027046768758832, threshold: 0.9335937500000011
swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/fold4: score: 0.7875062126996014, threshold: 0.8806640625000008
swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/fold5: score: 0.7503830381292295, threshold: 0.891992187500001

swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/fold1: score: 0.6496681029059532, threshold: 0.8906250000000009
swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/fold2: score: 0.7291322700543125, threshold: 0.9031250000000008
swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/fold3: score: 0.712814220702011, threshold: 0.927246093750001
swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/fold4: score: 0.7789634391449469, threshold: 0.8833007812500008
swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/fold5: score: 0.7445987861438901, threshold: 0.8957031250000009

ecaresnet26t_split3d3x12csn_l6_mixup_ep30/fold1: score: 0.6603261574289282, threshold: 0.901171875000001
ecaresnet26t_split3d3x12csn_l6_mixup_ep30/fold2: score: 0.7171293914568536, threshold: 0.9043945312500009
ecaresnet26t_split3d3x12csn_l6_mixup_ep30/fold3: score: 0.707478648308418, threshold: 0.9236328125000008
ecaresnet26t_split3d3x12csn_l6_mixup_ep30/fold4: score: 0.7661473342651401, threshold: 0.8882812500000008
ecaresnet26t_split3d3x12csn_l6_mixup_ep30/fold5: score: 0.742328060126813, threshold: 0.8926757812500009

ecaresnet26t_split3d2x15csn_l6_mixup_ep30/fold1: score: 0.6380428298638421, threshold: 0.9005859375000009
ecaresnet26t_split3d2x15csn_l6_mixup_ep30/fold2: score: 0.6968389137485008, threshold: 0.9002929687500009
ecaresnet26t_split3d2x15csn_l6_mixup_ep30/fold3: score: 0.6990990836925484, threshold: 0.929980468750001
ecaresnet26t_split3d2x15csn_l6_mixup_ep30/fold4: score: 0.7599457538682646, threshold: 0.8794921875000008
ecaresnet26t_split3d2x15csn_l6_mixup_ep30/fold5: score: 0.7311242069813252, threshold: 0.8962890625000011
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
        model = InkDetLightningModel.load_from_checkpoint(
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
