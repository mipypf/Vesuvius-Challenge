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
resnetrs50_split3d5x7csn_mixup_ep25/fold1: score: 0.6575029309805652, threshold: 0.896093750000001
resnetrs50_split3d5x7csn_mixup_ep25/fold3: score: 0.6890548603037959, threshold: 0.923925781250001

resnetrs50_split3d3x9csn_l6_mixup_ep25/fold1: score: 0.6630090422850091, threshold: 0.8995117187500009
resnetrs50_split3d3x9csn_l6_mixup_ep25/fold3: score: 0.6952362092376995, threshold: 0.9250000000000009

resnetrs50_split3d5x7csn_mixup_ep30/fold1: score: 0.661666662175347, threshold: 0.8962890625000011
resnetrs50_split3d5x7csn_mixup_ep30/fold2: score: 0.676702739882396, threshold: 0.896484375000001
resnetrs50_split3d5x7csn_mixup_ep30/fold3: score: 0.6893507333288211, threshold: 0.924218750000001
resnetrs50_split3d5x7csn_mixup_ep30/fold4: score: 0.7777119023923392, threshold: 0.8797851562500009
resnetrs50_split3d5x7csn_mixup_ep30/fold5: score: 0.7455737000141693, threshold: 0.8949218750000009

resnetrs50_split3d3x9csn_l6_mixup_ep30/fold1: score: 0.647394759423847, threshold: 0.8941406250000009
resnetrs50_split3d3x9csn_l6_mixup_ep30/fold2: score: 0.7227190397501271, threshold: 0.897070312500001
resnetrs50_split3d3x9csn_l6_mixup_ep30/fold3: score: 0.711266574379027, threshold: 0.9267578125000009
resnetrs50_split3d3x9csn_l6_mixup_ep30/fold4: score: 0.7807569334254353, threshold: 0.877734375000001
resnetrs50_split3d3x9csn_l6_mixup_ep30/fold5: score: 0.7286967877089129, threshold: 0.893066406250001

convnext_tiny_split3d5x7csn_mixup_ep30/fold1: score: 0.6480484068230761, threshold: 0.8995117187500009
convnext_tiny_split3d5x7csn_mixup_ep30/fold2: score: 0.7349025767720183, threshold: 0.901367187500001
convnext_tiny_split3d5x7csn_mixup_ep30/fold3: score: 0.6901466376748603, threshold: 0.930175781250001
convnext_tiny_split3d5x7csn_mixup_ep30/fold4: score: 0.78872785984954, threshold: 0.874414062500001
convnext_tiny_split3d5x7csn_mixup_ep30/fold5: score: 0.7579170092311271, threshold: 0.8947265625000009

convnext_tiny_split3d3x9csn_l6_mixup_ep30/fold1: score: 0.6609598722175213, threshold: 0.8945312500000009
convnext_tiny_split3d3x9csn_l6_mixup_ep30/fold2: score: 0.7103085807166868, threshold: 0.8946289062500009
convnext_tiny_split3d3x9csn_l6_mixup_ep30/fold3: score: 0.7002801042075558, threshold: 0.933203125000001
convnext_tiny_split3d3x9csn_l6_mixup_ep30/fold4: score: 0.7907143239234843, threshold: 0.8818359375000009
convnext_tiny_split3d3x9csn_l6_mixup_ep30/fold5: score: 0.7540930277816693, threshold: 0.893457031250001

swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/fold1: score: 0.6595671395781149, threshold: 0.894824218750001
swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/fold2: score: 0.7202787639366339, threshold: 0.8997070312500011
swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/fold3: score: 0.7005961531124031, threshold: 0.9301757812500011
swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/fold4: score: 0.7830941383181996, threshold: 0.8788085937500009
swinv2_tiny_window8_256_split3d5x7csn_mixup_ep30/fold5: score: 0.751246784061386, threshold: 0.897070312500001

swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/fold1: score: 0.649450337730675, threshold: 0.9015625000000009
swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/fold2: score: 0.7385365669462258, threshold: 0.9022460937500009
swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/fold3: score: 0.7156696272527835, threshold: 0.9275390625000011
swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/fold4: score: 0.7809394070653846, threshold: 0.8765625000000009
swinv2_tiny_window8_256_split3d3x9csn_l6_mixup_ep30/fold5: score: 0.7565042320219451, threshold: 0.8927734375000009

swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/fold1: score: 0.6437277482361012, threshold: 0.896875000000001
swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/fold2: score: 0.7162038860042182, threshold: 0.9078125000000009
swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/fold3: score: 0.7027386516443704, threshold: 0.9335937500000011
swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/fold4: score: 0.7874930884279626, threshold: 0.8802734375000008
swin_small_patch4_window7_224_split3d5x7csn_mixup_ep30/fold5: score: 0.7503897114445485, threshold: 0.8921875000000009

swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/fold1: score: 0.6496851695904378, threshold: 0.8904296875000008
swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/fold2: score: 0.729130118069193, threshold: 0.9031250000000008
swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/fold3: score: 0.7128404955412003, threshold: 0.927734375000001
swin_small_patch4_window7_224_split3d3x9csn_l6_mixup_ep30/fold4: score: 0.7789567471724264, threshold: 0.8835937500000008
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
                pred_batch = torch.sigmoid(model.model_ema.module(volume_new))

            for i in range(len(pred_batch)):
                if i % len(tta_set) == 1:
                    pred_batch[i] = pred_batch[i].flip(1)
                elif i % len(tta_set) == 2:
                    pred_batch[i] = pred_batch[i].flip(2)
                elif i % len(tta_set) == 3:
                    pred_batch[i] = pred_batch[i].flip(1).flip(2)

            pred_batch = F.interpolate(
                pred_batch.detach().to(torch.float32).cpu(),
                scale_factor=32,
                mode="bilinear",
                align_corners=True,
            ).numpy()
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
                count_pix_single = np.zeros_like(pi[0])
                if pad > 0:
                    count_pix_single[pad:-pad, pad:-pad] = np.ones_like(
                        pred_batch[0][0]
                    )
                else:
                    count_pix_single = np.ones_like(pred_batch[0][0])
                count_pix[
                    yi * 32 : yi * 32 + volume.shape[-2],
                    xi * 32 : xi * 32 + volume.shape[-1],
                ] += count_pix_single[:y_lim, :x_lim]
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
        np.save(f"{logdir}/logits_tta", p_valid)
        p_valid = p_valid > np.quantile(p_valid_tmp, threshold)
        score = fbeta_score(y_valid, p_valid, beta=0.5)
        print(f"{args.logdir}/fold{valid_idx}: score: {score}, threshold: {threshold}")
        np.save(
            f"{logdir}/preds_tta",
            (p_valid > np.quantile(p_valid_tmp, threshold)).astype(int),
        )
        np.save(f"{logdir}/true", y_valid)


if __name__ == "__main__":
    main(get_args())
